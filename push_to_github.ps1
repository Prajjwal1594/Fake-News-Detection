# Script to initialize, commit, and push Fake News Detection project to GitHub
Write-Host "Initializing Git Repository..." -ForegroundColor Cyan
git init

Write-Host "Adding remote origin https://github.com/prajjwal1594/FAKE-NEWS-DETECTION.git..." -ForegroundColor Cyan
git remote remove origin 2>$null
git remote add origin https://github.com/prajjwal1594/FAKE-NEWS-DETECTION.git

Write-Host "Staging all project files..." -ForegroundColor Cyan
git add .

Write-Host "Creating initial commit..." -ForegroundColor Cyan
git commit -m "Initial commit: Fake news detection application with ML models and Vercel deployment"

Write-Host "Renaming branch to main..." -ForegroundColor Cyan
git branch -M main

Write-Host "Pushing to GitHub repository (github.com/prajjwal1594/FAKE-NEWS-DETECTION)..." -ForegroundColor Cyan
git push -u origin main --force

Write-Host "Successfully pushed to GitHub!" -ForegroundColor Green
