#!/bin/bash
# Export YouTube Cookies Script
# 
# This script helps export fresh YouTube cookies from your browser
# and upload them to GCS for the backend to use.
#
# PREREQUISITES:
# 1. Install yt-dlp: brew install yt-dlp
# 2. Be logged into YouTube in Chrome/Firefox/Safari
# 3. Have gcloud CLI installed and authenticated
#
# USAGE: ./export-youtube-cookies.sh

set -e

COOKIES_FILE="youtube_cookies.txt"
GCS_BUCKET="cc-music-library"
GCS_PATH="secrets/youtube_cookies.txt"

echo "============================================"
echo "  YouTube Cookie Export Tool"
echo "============================================"
echo ""

# Check for yt-dlp
if ! command -v yt-dlp &> /dev/null; then
    echo "❌ yt-dlp not found. Install with: brew install yt-dlp"
    exit 1
fi

# Check for gcloud
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud not found. Install Google Cloud SDK first."
    exit 1
fi

echo "Choose your browser:"
echo "  1) Chrome"
echo "  2) Firefox"
echo "  3) Safari"
echo "  4) Brave"
echo "  5) Edge"
read -p "Enter choice (1-5): " browser_choice

case $browser_choice in
    1) BROWSER="chrome" ;;
    2) BROWSER="firefox" ;;
    3) BROWSER="safari" ;;
    4) BROWSER="brave" ;;
    5) BROWSER="edge" ;;
    *) echo "Invalid choice"; exit 1 ;;
esac

echo ""
echo "📦 Exporting cookies from $BROWSER..."

# Export cookies using yt-dlp
yt-dlp --cookies-from-browser "$BROWSER" --cookies "$COOKIES_FILE" --skip-download "https://www.youtube.com" 2>&1 || {
    echo "❌ Failed to export cookies. Make sure you're logged into YouTube in $BROWSER."
    exit 1
}

# Verify cookies were exported
if [ ! -f "$COOKIES_FILE" ]; then
    echo "❌ Cookie file not created"
    exit 1
fi

COOKIE_COUNT=$(grep -c "youtube.com" "$COOKIES_FILE" 2>/dev/null || echo "0")
echo "✅ Exported $COOKIE_COUNT YouTube cookies"

# Test the cookies
echo ""
echo "🧪 Testing cookies with a YouTube video..."
TEST_RESULT=$(yt-dlp --cookies "$COOKIES_FILE" -g "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 2>&1 | head -1)

if [[ "$TEST_RESULT" == http* ]]; then
    echo "✅ Cookies are working!"
else
    echo "⚠️  Cookie test returned: $TEST_RESULT"
    echo "   The cookies may still work for some videos."
fi

# Upload to GCS
echo ""
echo "☁️  Uploading cookies to GCS..."
gsutil cp "$COOKIES_FILE" "gs://$GCS_BUCKET/$GCS_PATH"

if [ $? -eq 0 ]; then
    echo "✅ Cookies uploaded to gs://$GCS_BUCKET/$GCS_PATH"
else
    echo "❌ Failed to upload cookies to GCS"
    exit 1
fi

# Trigger Cloud Run restart to pick up new cookies
echo ""
echo "🔄 Restarting Cloud Run service to pick up new cookies..."
gcloud run services update cc-music-pipeline --region us-central1 --no-traffic 2>/dev/null && \
gcloud run services update cc-music-pipeline --region us-central1 --to-latest 2>/dev/null || {
    echo "⚠️  Could not restart Cloud Run. You may need to redeploy manually."
}

echo ""
echo "============================================"
echo "  ✅ Cookie export complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Wait 1-2 minutes for Cloud Run to restart"
echo "  2. Test with: curl -X POST https://cc-music-pipeline-owq2vk3wya-uc.a.run.app/api/batch/submit ..."
echo ""

# Clean up local cookie file
rm -f "$COOKIES_FILE"

