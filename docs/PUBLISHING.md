# Publishing this prepared repository

The cookbook is a standalone Git repository with branch `main`. It contains no remote configuration. Set your desired GitHub owner and visibility when publishing.

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

Choose `--private` instead if desired. Alternatively create an empty repository in the GitHub UI, add its Git URL as `origin`, and push `main`. The preparation work does not create a GitHub repository or send any files to GitHub.

Keep config.json, downloaded model/book data, local results, compiler caches and extension binaries out of commits. `.gitignore` covers the standard local paths. If changing the image or overlays, update the manifest and validation evidence together; don't advertise the historical performance as a measurement of an untested revision.
