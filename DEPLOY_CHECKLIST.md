# Deploy Checklist

Before pushing or deploying this project:

- [ ] Repository must be Private.
- [ ] Review `.gitignore` before push.
- [ ] Do not push secrets, passwords, API keys, tokens, or `.env` files.
- [ ] Do not push company data or company-only files.
- [ ] Test locally before deploy.
- [ ] Confirm the dashboard runs with `py -m streamlit run app.py`.
- [ ] Confirm `py -m py_compile app.py analyzer.py predictor.py` passes.
- [ ] Confirm required local files exist in `data/` and `output/`.
- [ ] If deploying publicly, add login/password protection before publishing.
- [ ] Do not connect this app to a company server without a separate approved deployment plan.

Safety note:

Historical statistical analysis only. This system does not predict or guarantee lottery outcomes.
