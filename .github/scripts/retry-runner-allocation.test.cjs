const assert = require('node:assert/strict');
const {test} = require('node:test');
const recover = require('./retry-runner-allocation.cjs');

function fixture() {
  const run = {
    id: 42, path: '.github/workflows/build-wheels-upload-pypi.yml',
    event: 'push', head_branch: 'v1.5.3', head_sha: 'release-sha',
    head_repository: {full_name: 'owner/sep'}, run_attempt: 3,
    status: 'completed', conclusion: 'failure',
  };
  const job = {
    id: 7, name: 'macOS', status: 'completed', conclusion: 'cancelled',
    steps: [], check_run_url: 'https://api.github.com/repos/owner/sep/check-runs/8',
  };
  const f = {
    run, jobs: [job], annotations: [{message:
      'The job was not acquired by Runner of type hosted even after multiple attempts'}],
    currentSha: 'release-sha', requests: [], reruns: [],
  };
  const actions = {
    getWorkflowRun: async args => { f.requests.push(args); return {data: f.run}; },
    listWorkflowRunsForRepo: Symbol('runs'),
    listJobsForWorkflowRunAttempt: Symbol('jobs'),
    reRunWorkflowFailedJobs: async args => { f.reruns.push(args); },
  };
  const checks = {listAnnotations: Symbol('annotations')};
  f.github = {
    rest: {actions, checks, repos: {
      getCommit: async () => ({data: {sha: f.currentSha}}),
    }},
    paginate: async (endpoint, args) => {
      f.requests.push(args);
      if (endpoint === actions.listWorkflowRunsForRepo) return [f.run];
      if (endpoint === actions.listJobsForWorkflowRunAttempt) return f.jobs;
      if (endpoint === checks.listAnnotations) return f.annotations;
      throw new Error('Unexpected API');
    },
  };
  f.context = {repo: {owner: 'owner', repo: 'sep'}, eventName: 'workflow_run',
    payload: {workflow_run: {...run}}};
  f.core = {info() {}, notice() {}};
  return f;
}

test('retry allocation-only failures and skipped dependents, reusing successful jobs', async () => {
  const f = fixture();
  f.jobs.push({status: 'completed', conclusion: 'success', steps: [{name: 'tests'}]});
  f.jobs.push({status: 'completed', conclusion: 'skipped'});
  await recover(f);
  assert.deepEqual(f.reruns, [{owner: 'owner', repo: 'sep', run_id: 42}]);
  assert.ok(f.requests.some(r => r.attempt_number === 3));
  assert.ok(f.requests.some(r => r.check_run_id === 8));
});

for (const change of ['executed steps', 'test failure', 'manual cancellation',
                      'missing check', 'mixed failures', 'no failures']) {
  test(`do not retry ${change}`, async () => {
    const f = fixture();
    if (change === 'executed steps') f.jobs[0].steps = [{name: 'Run tests'}];
    if (change === 'test failure') f.annotations = [{message: 'Tests failed'}];
    if (change === 'manual cancellation') f.annotations = [];
    if (change === 'missing check') delete f.jobs[0].check_run_url;
    if (change === 'mixed failures') f.jobs.push({status: 'completed',
      conclusion: 'failure', steps: [{name: 'Build wheel'}]});
    if (change === 'no failures') f.jobs = [{conclusion: 'success'}];
    await recover(f);
    assert.equal(f.reruns.length, 0);
  });
}

for (const change of ['success', 'active run', 'stale event', 'obsolete ref',
                      'foreign repository', 'pull request', 'unrelated workflow']) {
  test(`ignore ${change}`, async () => {
    const f = fixture();
    if (change === 'success') f.run.conclusion = 'success';
    if (change === 'active run') f.run.status = 'queued';
    if (change === 'stale event') f.run.run_attempt++;
    if (change === 'obsolete ref') f.currentSha = 'new-sha';
    if (change === 'foreign repository') f.run.head_repository.full_name = 'fork/sep';
    if (change === 'pull request') f.run.event = 'pull_request';
    if (change === 'unrelated workflow') f.run.path = '.github/workflows/other.yml';
    await recover(f);
    assert.equal(f.reruns.length, 0);
  });
}

test('scheduled checks recover the existing tag run without changing its ref', async () => {
  const f = fixture();
  f.context.eventName = 'schedule';
  await recover({...f, now: new Date('2026-11-13T12:00:00Z')});
  assert.ok(f.requests.some(r => r.created === '>=2026-10-15T12:00:00.000Z'));
  assert.equal(f.reruns.length, 1);
  assert.equal(f.run.head_branch, 'v1.5.3');
});

test('manual bootstrap inspects the selected run', async () => {
  const f = fixture();
  f.context.eventName = 'workflow_dispatch';
  await recover({...f, runId: '42'});
  assert.equal(f.reruns.length, 1);
  await assert.rejects(recover({...f, runId: '42; shell command'}), /Invalid run ID/);
});

test('deleted refs are ignored, but unexpected API errors fail visibly', async () => {
  const f = fixture();
  f.github.rest.repos.getCommit = async () => { throw Object.assign(new Error(), {status: 404}); };
  await recover(f);
  assert.equal(f.reruns.length, 0);
  f.github.rest.repos.getCommit = async () => { throw new Error('API unavailable'); };
  await assert.rejects(recover(f), /API unavailable/);
});

test('a concurrent retry conflict is harmless', async () => {
  const f = fixture();
  f.github.rest.actions.reRunWorkflowFailedJobs = async () => {
    throw Object.assign(new Error('already queued'), {status: 409});
  };
  await recover(f);
});
