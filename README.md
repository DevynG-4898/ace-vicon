# ACE – Vicon Tennis Serve Analysis

ACE is a motion capture–based tennis serve analysis system built using Vicon.  
The system compares a user's serve to a reference player and provides quantitative feedback to help the user match the model’s motion.

---

## Project Overview

This project uses Vicon motion capture data to:

- Record a reference tennis serve
- Record a user serve
- Load and process 3D motion data
- Compute motion metrics
- Compare user motion against a reference model

The long-term goal is to provide actionable feedback for improving serve mechanics.

---

## Repository Structure

ace-vicon/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── src/
│ ├── data/
│ ├── analysis/ (will be added later)
│ ├── visualization/ (will be added later)
│ └── main.py
│
├── tests/
├── docs/
├── requirements.txt
├── README.md
├── .gitignore
├── setup.md
│
│
├── plotting/
├── markers/
│ ├── serve1/
│ ├── unmarked/
├── load_data.py/
├── plot.py/
├── requirements.txt/

---

## How To Run

1. Follow instructions in `setup.md`
2. Activate your virtual environment
3. From the project root, run:

## How To Run Plot

1. cd .\plotting\
2. python plot.py

---

## Current Sprint Goal

Sprint 1 focuses on:

- Recording reference serve data
- Exporting Vicon CSV files
- Building a working data loader

---

## Team

Project ACE

- Allison Turnbow
- Max Gavin
- Biplav Adhikari
- Devyn Gayle
- Maximiliano Barajas
- Jaime Favela

---

## Tools

- Vicon Motion Capture
- Python
- Pandas / NumPy
- Matplotlib
- Jira (project tracking)
- Overleaf (documentation)
