#!/bin/bash
# Export YouTube Cookies for Local Pipeline
#
# This exports fresh YouTube cookies from your browser for use
# with the local training pipeline.
#
# PREREQUISITES:
# 1. Be logged into YouTube in your browser
# 2. yt-dlp installed (brew install yt-dlp)
#
# USAGE: ./export-cookies-local.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COOKIES_FILE="$SCRIPT_DIR/cookies.txt"

echo "============================================"
echo "  YouTube Cookie Export (Local)"
echo "============================================"
echo ""

# Check for yt-dlp
if ! command -v yt-dlp &> /dev/null; then
    echo "❌ yt-dlp not found. Install with: brew install yt-dlp"
    exit 1
fi

echo "Choose your browser:"
echo "  1) Chrome"
echo "  2) Firefox"
echo "  3) Safari"
echo "  4) Brave"
echo ""
read -p "Enter choice (1-4): " browser_choice

case $browser_choice in
    1) BROWSER="chrome" ;;
    2) BROWSER="firefox" ;;
    3) BROWSER="safari" ;;
    4) BROWSER="brave" ;;
    *) echo "Invalid choice"; exit 1 ;;
esac

echo ""
echo "📦 Exporting cookies from $BROWSER..."
echo "   (You may see a keychain prompt - allow access)"
echo ""

# Export cookies using yt-dlp
yt-dlp --cookies-from-browser "$BROWSER" --cookies "$COOKIES_FILE" --skip-download "https://www.youtube.com" 2>&1 || {
    echo ""
    echo "❌ Failed to export cookies."
    echo ""
    echo "Troubleshooting:"
    echo "  1. Make sure you're logged into YouTube in $BROWSER"
    echo "  2. Try closing the browser and running again"
    echo "  3. If using Chrome, you may need to allow keychain access"
    echo ""
    exit 1
}

# Verify cookies
if [ ! -f "$COOKIES_FILE" ]; then
    echo "❌ Cookie file not created"
    exit 1
fi

COOKIE_COUNT=$(grep -c "youtube.com" "$COOKIES_FILE" 2>/dev/null || echo "0")
echo "✅ Exported $COOKIE_COUNT YouTube cookies"
echo "   Saved to: $COOKIES_FILE"

# Test the cookies
echo ""
echo "🧪 Testing cookies..."
TEST_VIDEO="https://www.youtube.com/watch?v=xsUrdpKD5wM"  # N'Ko video
TEST_RESULT=$(yt-dlp --cookies "$COOKIES_FILE" -f best -g "$TEST_VIDEO" 2>&1 | head -1)

if [[ "$TEST_RESULT" == http* ]]; then
    echo "✅ Cookies are working!"
    echo ""
    echo "============================================"
    echo "  Cookie export successful!"
    echo "============================================"
    echo ""
    echo "Now run the pipeline:"
    echo "  python run_extraction.py --retry-failed"
    echo ""
else
    echo "⚠️  Cookie test failed: $TEST_RESULT"
    echo ""
    echo "The cookies may not work. Try:"
    echo "  1. Log out and log back into YouTube"
    echo "  2. Clear browser cookies and log in fresh"
    echo "  3. Try a different browser"
    echo ""
fi

