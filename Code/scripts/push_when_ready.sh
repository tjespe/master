#!/bin/bash
git pull --no-edit
git add .
git commit -m "Add preds"
git push
sleep 10
$0
