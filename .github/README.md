# CI and releases

Branch pushes and pull requests run `CI`. Release tags (`v*`) run
`build-wheels-upload-pypi`, which calls the same test matrix before building
the source distribution and wheels. Only a tag push can upload to PyPI;
manual builds do not publish.

For a release, commit the `CHANGES.md` release heading, tag that commit, and
push the branch and tag. Changelog-only branch pushes skip CI, so this creates
one release run containing tests, builds, and publication. If the release commit
also changes code, the branch push runs CI as well.

The runner-allocation recovery workflow retries failed jobs in the original
run, preserving its tag-push event and successful jobs.
