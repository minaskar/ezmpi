#!/bin/bash
# Script to prepare a release

set -e

VERSION=$1

# Check if version provided
if [ -z "$VERSION" ]; then
    echo "Usage: ./prepare-release.sh <version>"
    echo "Example: ./prepare-release.sh 1.2.3"
    exit 1
fi

# Validate version format
if ! [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Invalid version format. Use X.Y.Z"
    exit 1
fi

echo "Preparing release $VERSION..."

# Update version in __init__.py
if [ -f "src/ezmpi/__init__.py" ]; then
    sed -i.bak "s/__version__ = \".*\"/__version__ = \"$VERSION\"/g" src/ezmpi/__init__.py
    rm src/ezmpi/__init__.py.bak
    echo "✓ Updated version in src/ezmpi/__init__.py"
else
    echo "✗ src/ezmpi/__init__.py not found"
    exit 1
fi

# Update CHANGELOG.md if it exists
if [ -f "CHANGELOG.md" ]; then
    # Check if version already exists in changelog
    if grep -q "## \[$VERSION\]" CHANGELOG.md; then
        echo "✓ Version $VERSION already exists in CHANGELOG.md"
    else
        # Add new version section at the top
        TODAY=$(date +%Y-%m-%d)
        sed -i.bak "2a\\
## [$VERSION] - $TODAY\\
\\
### Added\\
-\\
\\
### Changed\\
-\\
\\
### Fixed\\
-\\
" CHANGELOG.md
        rm CHANGELOG.md.bak
        echo "✓ Added $VERSION section to CHANGELOG.md"
    fi
else
    echo "✗ CHANGELOG.md not found"
    exit 1
fi

# Check if we're in a git repository
if [ -d ".git" ]; then
    # Check if there are any changes to commit
    if ! git diff --quiet; then
        git add src/ezmpi/__init__.py CHANGELOG.md
        git commit -m "Prepare release v$VERSION"
        echo "✓ Committed version update"
    else
        echo "✓ No changes to commit"
    fi
    
    # Create and push tag
    git tag -a "v$VERSION" -m "Release version $VERSION"
    echo "✓ Created tag v$VERSION"
    
    # Ask about pushing
    echo "Ready to push. Run: git push origin main --tags"
    echo "This will trigger the release workflow."
else
    echo "✗ Not in a git repository"
    exit 1
fi

echo ""
echo "Release v$VERSION prepared successfully!"
echo "Next steps:"
echo "1. Review changes: git log --oneline -3"
echo "2. Push to trigger release: git push origin main --tags"