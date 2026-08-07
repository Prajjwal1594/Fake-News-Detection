# Script to train high-accuracy models, commit, and push Fake News Detection project to GitHub

Write-Host "Training High-Accuracy Machine Learning Models..." -ForegroundColor Yellow
python scripts/train_models.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Model training failed! Please check your Python environment." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Initializing Git Repository..." -ForegroundColor Cyan
git init

Write-Host "Adding remote origin https://github.com/prajjwal1594/Fake-News-Detection.git..." -ForegroundColor Cyan
git remote remove origin 2>$null
git remote add origin https://github.com/prajjwal1594/Fake-News-Detection.git

Write-Host "Staging all project files and trained models..." -ForegroundColor Cyan
git add .

Write-Host "Creating commit..." -ForegroundColor Cyan
git commit -m "Optimize models with FeatureUnion word+char TF-IDF, ComplementNB, ExtraTrees, and weighted Voting Ensemble for maximum accuracy"

Write-Host "Renaming branch to main..." -ForegroundColor Cyan
git branch -M main

Write-Host "Pushing to GitHub repository (github.com/prajjwal1594/Fake-News-Detection)..." -ForegroundColor Cyan
git push -u origin main --force

Write-Host "Successfully trained models and pushed to GitHub!" -ForegroundColor Green
