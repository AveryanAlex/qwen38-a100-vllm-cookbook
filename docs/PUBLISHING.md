# Publishing this prepared repository

The cookbook is published at [AveryanAlex/qwen38-a100-vllm-cookbook](https://github.com/AveryanAlex/qwen38-a100-vllm-cookbook), with default branch `main` and public visibility. The instructions below also describe publishing a separately prepared copy under your own account.

## Publication review, 2026-09-11

The release review passed 15 launcher/replay tests, exact reconstruction of every source patch, and repository checks for syntax, overlay/measurement/book hashes, documentation links and known private paths. A separate scan of all 272 then-reachable historical file blobs found no matches for private machine paths/addresses or the checked credential patterns (private keys, GitHub/Hugging Face tokens and AWS access-key IDs). This is a scoped scan, not a guarantee against every possible secret format. The tracked tree is approximately 8.2 MB and contains no model weights or compiled extensions.

The review fixed historical replay inheriting the new 512K/CPU-cache settings from the local config, and updated the host-memory requirements, overlay count and book/patch attribution. Historical commands now retain their recorded native context and cache settings even when the current serving config selects 512K and 128 GiB offloading.

The published validation scope remains explicit: 512K execution and eight-code retrieval succeeded but strict ordering failed; some performance measurements had other traffic; the portable service's lifecycle was tested with a disposable CPU container rather than a complete GPU redeployment. These limitations are recorded in the context and validation guides. Publishing does not change their status.

Before the first push:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/check_repository.py
python3 scripts/verify_patches.py
git status --short
git diff --cached --stat
```

Review the files and commit them if not already committed:

```bash
git add README.md LICENSE NOTICE .gitignore .gitattributes .github config.example.json manifest.json docs overlays patches kernels benchmarks scripts tests measurements
git commit -m "Add reproducible Qwen3.8 A100 vLLM cookbook"
```

With the GitHub CLI authenticated, create and publish the repository:

```bash
gh repo create qwen38-a100-vllm-cookbook --public --source=. --remote=origin --push
```

For a separately prepared copy, choose `--private` instead if desired. Alternatively create an empty repository in the GitHub UI, add its Git URL as `origin`, and push `main`. For this repository, `origin` is already configured; subsequent committed updates use `git push origin main`.

Keep config.json, downloaded model/book data, local results, compiler caches and extension binaries out of commits. `.gitignore` covers the standard local paths. If changing the image or overlays, update the manifest and validation evidence together; don't advertise the historical performance as a measurement of an untested revision.
