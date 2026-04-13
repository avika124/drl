# GCP Credentials Rotation Guide

**Security:** If `credentials.json` was ever committed or exposed, rotate the key immediately.

## Step 1: Create a New Key in GCP Console

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select project: **ad-metrics-pipeline**
3. Navigate to **IAM & Admin** → **Service Accounts**
4. Find: `ad-metrics-bot@ad-metrics-pipeline.iam.gserviceaccount.com`
5. Click the service account → **Keys** tab
6. Click **Add Key** → **Create new key** → **JSON**
7. Download the new JSON file

## Step 2: Replace the Old Credentials

1. Save the new JSON file as `credentials.json` in this directory (or another secure location)
2. **Delete the old key** in GCP Console:
   - Service Account → Keys → find the old key → Delete
   - This invalidates the old credentials immediately

## Step 3: Use Environment Variable (Recommended)

Instead of hardcoding the path, set the environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"
```

Or add to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcp/ad-metrics-credentials.json"
```

## Step 4: Verify

```bash
# Test BigQuery access
python -c "
from google.cloud import bigquery
client = bigquery.Client(project='ad-metrics-pipeline')
print('Credentials OK:', client.project)
"
```

## Notes

- `credentials.json` is in `.gitignore` — never commit it
- All DRL scripts now read from `GOOGLE_APPLICATION_CREDENTIALS` env var first
- Fallback to `credentials.json` in current directory if env var is not set
