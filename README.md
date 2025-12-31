<p align="center">
  <img width="140" alt="BrainWave Logo" src="https://github.com/user-attachments/assets/37e5893d-a458-417b-b3b3-4b766e01437c" alt="BrainWave logo" />

</p>

<h1 align="center">BrainWave</h1>

<p align="center">
  <b>Binaural beats audio generator + information center</b><br/>
  Self-hosted. Tons of features. Beautiful UI.
</p>

<p align="center">
  <a href="https://github.com/jlar0che/BrainWave/releases"><img alt="Release" src="https://img.shields.io/github/v/release/jlar0che/BrainWave?sort=semver"></a>
  <a href="https://hub.docker.com/r/jlaroche/brainwave"><img alt="Docker pulls" src="https://img.shields.io/docker/pulls/jlaroche/brainwave"></a>
  <a href="https://hub.docker.com/r/jlaroche/brainwave"><img alt="Docker image size" src="https://img.shields.io/docker/image-size/jlaroche/brainwave/latest"></a>
  <a href="LICENSE"><img alt="License: GPL-3.0" src="https://img.shields.io/badge/license-GPL--3.0-blue"></a>
</p>

<p align="center">
<img src="https://github.com/user-attachments/assets/e11d969f-2b64-4100-9971-4ee7190f3551" alt="BrainWave main UI" />
</p>

---

## 🎧 What are binaural beats?

A binaural beat is an auditory illusion created when you listen to two nearby frequencies, one in each ear (headphones required).
Your brain perceives a third “beat” at the difference between them.

<b>Example:</b>
<ul>
<li><b>Left ear:</b> 396 Hz</li>
<li><b>Right ear:</b> 404 Hz</li>
<li><b>Perceived beat:</b> 8 Hz</li>
</ul>

BrainWave lets you choose a carrier frequency (the tone you hear) and a beat frequency (the difference between left/right).

---

## ✨ What makes BrainWave different?
<ul>
<li><b>Live Scope:</b> real-time waveform view while audio plays</li>
<li><b>Realtime brainwave band readout:</b> see the current band label (Delta / Theta / Alpha / Beta / Gamma)</li>
<li><b>Export audio:</b> render your session to files (MP3 / OGG / FLAC)</li>
<li><b>Presets:</b> save/recall sessions, import/export BrainWave Preset files (.bwp)</li>
<li><b>Library:</b> curated reading list + bundled public-domain docs (PDFs) inside the UI</li>
<li><b>Modern Architecture:</b> self-hosted, web accessible and optimized for mobile</li>
</ul>

---

## ⚡ Quickstart (Docker)

**Clone the repository**:

```bash
git clone https://github.com/jlar0che/BrainWave.git
cd BrainWave
```

Open the `docker-compose.yml` file:

```yaml
services:
  brainwave:
    image: jlaroche/brainwave:latest
    container_name: brainwave
    ports:
      - "5890:5890"
    environment:
      # Set a real secret for production (at least 32 bytes / 64 hex chars)
      - SECRET_KEY=change-me-to-a-long-random-string

      # Optional: upload size (bytes)
      - MAX_UPLOAD_BYTES=65536

    volumes:
      # Persist exports + presets + library data
      - ./static/output:/app/static/output
      - ./static/presets:/app/static/presets
      - ./static/data:/app/static/data
      - ./logs:/app/logs
      - ./static/docs/public_domain:/app/static/docs/public_domain

    restart: unless-stopped
```

**Set up your environment variable(s) in the docker-compose.yml file**:
```yaml
environment:
      # Set a real secret for production (at least 32 bytes / 64 hex chars)
      - SECRET_KEY=change-me-to-a-long-random-string

      # Optional: upload size (bytes)
      - MAX_UPLOAD_BYTES=65536
```

<b>NOTE:</b>
You can easily create a secret key at https://it-tools.tech/token-generator

---

## 📸 Screenshots

<p align="center">
<img src="https://github.com/user-attachments/assets/b1b622ef-6819-4e0c-8d24-f5b248a377ca" alt="BrainWave Screenshot 1" />
</p>

<p align="center">
<img src="https://github.com/user-attachments/assets/5c4ee6f3-1da8-4639-9c85-cbdb51117728" alt="BrainWave Screenshot 2" />
</p>

<p align="center">
<img src="https://github.com/user-attachments/assets/bbd40a95-4bbb-43f2-bd55-b83165080590" alt="BrainWave Screenshot 3" />
</p>

<p align="center">
<img src="https://github.com/user-attachments/assets/35575d54-ef73-4a84-8380-3188f571e104" alt="BrainWave Screenshot 4" />
</p>

<p align="center">
<img src="https://github.com/user-attachments/assets/a18a07ef-4ea3-4754-be09-fe3b424b0fda" alt="BrainWave Screenshot 5" />
</p>

---

## 🤝 Contributing

PRs welcome — especially for:
<ul>
<li>more presets/templates</li>
<li>additional relevant resources in the Vault/Library</li>
<li>UI improvements and accessibility tweaks</li>
<li>additional features</li>
</ul>
