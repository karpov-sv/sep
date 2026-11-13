const WORKFLOWS = new Set([
  '.github/workflows/python-package-tox.yml',
  '.github/workflows/build-wheels-upload-pypi.yml',
]);
const ALLOCATION_ERROR = 'The job was not acquired by Runner of type hosted';

async function recover({github, context, core, runId = '', now = new Date()}) {
  const repo = context.repo;
  let candidates;
  if (runId) {
    if (!/^[1-9][0-9]*$/.test(runId)) throw new Error('Invalid run ID');
    candidates = [{id: Number(runId)}];
  } else {
    // Revisit all eligible failures on completion events too. GitHub may
    // publish allocation annotations after the completion event, and cron
    // checks can be delayed. Another run's completion can then recover it.
    // GitHub permits reruns for 30 days. Keep recovering current branch/tag
    // runs throughout that window, without repeatedly running obsolete code.
    const since = new Date(now.getTime() - 29 * 86400000).toISOString();
    candidates = await github.paginate(github.rest.actions.listWorkflowRunsForRepo, {
      ...repo, status: 'failure', created: `>=${since}`, per_page: 100,
    });
  }

  for (const candidate of candidates) {
    // Re-read live state: completion events can arrive after another retry.
    const {data: run} = await github.rest.actions.getWorkflowRun({
      ...repo, run_id: candidate.id,
    });
    if (!WORKFLOWS.has(run.path) || run.status !== 'completed' ||
        run.conclusion !== 'failure' ||
        !['push', 'workflow_dispatch'].includes(run.event) ||
        run.head_repository?.full_name !== `${repo.owner}/${repo.repo}` ||
        !run.head_branch ||
        (candidate.run_attempt && candidate.run_attempt !== run.run_attempt)) {
      core.info(`Run ${run.id}: not eligible for recovery.`);
      continue;
    }

    let current;
    try {
      current = (await github.rest.repos.getCommit({...repo, ref: run.head_branch})).data;
    } catch (error) {
      if (error.status !== 404) throw error;
      core.info(`Run ${run.id}: branch/tag no longer exists.`);
      continue;
    }
    if (current.sha !== run.head_sha) {
      core.info(`Run ${run.id}: branch/tag now points to a different commit.`);
      continue;
    }

    const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRunAttempt, {
      ...repo, run_id: run.id, attempt_number: run.run_attempt, per_page: 100,
    });
    const unsuccessful = jobs.filter(job =>
      !['success', 'skipped'].includes(job.conclusion));
    if (!unsuccessful.length) continue;

    let allocationOnly = true;
    for (const job of unsuccessful) {
      if (job.status !== 'completed' || job.steps?.length ||
          !['cancelled', 'failure', 'timed_out'].includes(job.conclusion)) {
        allocationOnly = false;
        break;
      }
      const match = /\/check-runs\/(\d+)$/.exec(job.check_run_url || '');
      if (!match) {
        allocationOnly = false;
        break;
      }
      const annotations = await github.paginate(github.rest.checks.listAnnotations, {
        ...repo, check_run_id: Number(match[1]), per_page: 100,
      });
      if (!annotations.some(item => item.message?.includes(ALLOCATION_ERROR))) {
        allocationOnly = false;
        break;
      }
    }
    if (!allocationOnly) {
      core.info(`Run ${run.id}: not every unsuccessful job has a confirmed allocation failure; leaving it unchanged.`);
      continue;
    }

    // Preserve the original ref/event (including tag-push publishing), and
    // reuse successful jobs. GitHub also reruns skipped dependent jobs.
    try {
      await github.rest.actions.reRunWorkflowFailedJobs({...repo, run_id: run.id});
      core.notice(`Run ${run.id}: retrying runner allocation failures from attempt ${run.run_attempt}.`);
    } catch (error) {
      if (error.status !== 409) throw error;
      core.info(`Run ${run.id}: another retry is already in progress.`);
    }
  }
}

module.exports = recover;
