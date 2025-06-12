# oc_py

oc_py is a lightweight Python package for loading, processing, modeling and visualizing Eclipse Timing Variations (O–C) data from eclipsing binaries. It supports linear, quadratic, Keplerian (LiTE), and full N-body fits (via REBOUND), plus MCMC uncertainty estimation (via emcee).

---

## 🚀 Features

- **Data I/O & Preprocessing**  
  – Read O–C tables in CSV, plain text 
  – Automatic epoch correction and error filtering  

- **Built-in Model Components**  
  – **Lin:** linear period drift  
  – **Quad:** quadratic (secular) period change  
  – **LiTE:** Light-Time Effect Keplerian solution  
  – **Newtonian:** full N-body integration via REBOUND  

- **Fitting & Uncertainty**  
  – Least-squares optimization (Levenberg–Marquardt)  
  – Bayesian MCMC sampling with emcee  
  – Compute χ², reduced χ², AIC, BIC  

- **Output & Visualization**  
  – Publication-quality plots (Matplotlib)  
  – Corner plots for posterior distributions  
---
