# N'Ko Continuous Learning Pipeline - Deployment Guide

Deploy the vocabulary enrichment and video processing pipeline to a server.

## Quick Deploy Options

### Option 1: Railway (Recommended - Easiest)

1. **Push to GitHub** (if not already)
2. **Go to [railway.app](https://railway.app)**
3. **Create new project** → Deploy from GitHub repo
4. **Set root directory** to `training/`
5. **Add environment variables**:
   ```
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_SERVICE_KEY=your-service-role-key
   GEMINI_API_KEY=your-gemini-api-key
   ```
6. **Deploy** - Railway will auto-detect Docker

### Option 2: Render

1. Go to [render.com](https://render.com)
2. Create **Background Worker** service
3. Connect GitHub repo, set `training/` as root
4. Add environment variables
5. Deploy

### Option 3: Fly.io

```bash
cd training/

# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Create app
fly launch --name nko-learning

# Set secrets
fly secrets set SUPABASE_URL="https://your-project.supabase.co"
fly secrets set SUPABASE_SERVICE_KEY="your-key"
fly secrets set GEMINI_API_KEY="your-key"

# Deploy
fly deploy
```

### Option 4: Docker on VPS (DigitalOcean, Linode, etc.)

```bash
# SSH to your server
ssh user@your-server

# Clone repo
git clone https://github.com/your-username/learnnko.git
cd learnnko/training

# Create .env file
cat > .env << EOF
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
GEMINI_API_KEY=your-gemini-api-key
EOF

# Run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f enrichment
```

### Option 5: Google Cloud Run (Jobs)

```bash
cd training/

# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT/nko-learning

# Create Cloud Run Job (runs continuously)
gcloud run jobs create nko-enrichment \
  --image gcr.io/YOUR_PROJECT/nko-learning \
  --set-env-vars SUPABASE_URL="..." \
  --set-secrets SUPABASE_SERVICE_KEY=supabase-key:latest \
  --set-secrets GEMINI_API_KEY=gemini-key:latest \
  --region us-central1

# Execute job
gcloud run jobs execute nko-enrichment
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Yes | Supabase service role key |
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `ENRICHMENT_INTERVAL` | No | Seconds between batches (default: 300) |
| `BATCH_SIZE` | No | Words per batch (default: 10) |

---

## Services

### Enrichment (Always Running)
Continuously enriches vocabulary using dictionary lookups and AI.

```bash
# Start
docker-compose up -d enrichment

# Logs
docker-compose logs -f enrichment
```

### Video Extraction (On-Demand)
Process videos from YouTube channels.

```bash
# Run once
docker-compose --profile extraction up extraction

# Or run specific channel
docker-compose run extraction python scripts/run_extraction.py --channel ankataa --limit 50
```

### Forum Scraper (Periodic)
Import knowledge from Ankataa forum.

```bash
docker-compose --profile forum up forum-scraper
```

---

## Monitoring

### Check Status
```bash
# SSH to server
docker-compose exec enrichment python scripts/scheduled_exploration.py --status
```

### View Logs
```bash
docker-compose logs -f --tail=100 enrichment
```

### Restart Service
```bash
docker-compose restart enrichment
```

---

## Cost Estimates

| Service | Monthly Cost |
|---------|--------------|
| Railway (Hobby) | $5/month |
| Render (Starter) | $7/month |
| Fly.io (1 shared CPU) | ~$5/month |
| DigitalOcean Droplet | $6/month |
| Cloud Run (continuous) | ~$10-20/month |

Gemini API costs depend on usage (~$0.0001/word for enrichment).

