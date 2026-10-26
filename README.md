# Cognitive FHSS Link Monitor (cloud dashboard)

A small web app that shows a cognitive frequency-hopping link in real time: link state from the
receiver's Edge AI classifier, the channel in use, packet loss for the cognitive link against a
static FHSS link, and every hop command. It runs from the simulator today and from the real
hardware later, using the same database and the same page.

All values in replay mode are **simulated**, not measured.

## Run it on your computer
    python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    python app.py
Open http://localhost:8000, choose a scenario and click **Start replay**.

Files: `app.py` (web server and API), `db.py` (SQLite storage), `replay.py` (streams the simulation
into the database), `fhss_sim.py` (the simulator), `classifier_weights.npz` (small trained model),
`templates/index.html` (the dashboard). To retrain the stand-in model: `pip install scikit-learn`
then `python train_weights.py`.

## Deploy on Huawei Cloud (ECS)
Console menus change, so follow the names loosely.
1. Create an **ECS** instance (Ubuntu 22.04, the smallest general-purpose size is enough) and bind an
   elastic public IP.
2. In the instance's **security group**, add an inbound rule for TCP port 8000 (or 80 if you put a proxy
   in front). Keep port 22 restricted to your own IP.
3. Connect by SSH and install: `sudo apt update && sudo apt install -y python3-venv`.
4. Copy this folder to the server (scp, git clone from your repository, or the console's file upload).
5. On the server:
       cd fhss-dashboard && python3 -m venv venv && source venv/bin/activate
       pip install -r requirements.txt
       gunicorn -w 1 --threads 8 -b 0.0.0.0:8000 app:app
   Use one worker (the replay runs inside the server process).
6. Open `http://<your-ECS-public-IP>:8000`.
7. To keep it running after you log out, run it under `systemd` or `nohup`.

**Using more Huawei Cloud services (recommended for the competition):** replace SQLite in `db.py`
with an **RDS** (MySQL) database, keeping the same table layout; store the project's dataset or
logs in **OBS**; train or retrain the classifier on **ModelArts**. Check the competition rules for
the services they expect.

## Sending real data from the hardware
The receiver (or a laptop reading its serial port) posts JSON every 100 ms:

    POST /api/ingest
    {"new_run": true, "duration": 60, "bins": [
      {"t": 0.1, "cog_ch": 12, "state": 0, "p_jam": 0.02, "sent_cog": 50, "recv_cog": 50}]}

`state` is 0 clear, 1 interfered, 2 jammed. Hop events:

    POST /api/hop
    {"t": 12.4, "from_ch": 12, "to_ch": 20, "detect_ms": 8, "delay_ms": 80}

Because a jammer on the 2.4 GHz band can also disturb the ESP32's own Wi-Fi, buffer readings on the
board or send them through a laptop over USB so the dashboard keeps updating during a jamming test.
There is no login on this app. Do not leave it open on the internet with real data.
