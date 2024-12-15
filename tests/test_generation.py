import subprocess
import os

def test_generation():
    # Before running this test, ensure you have trained models or placeholders.
    config = "configs/config.yaml"
    vae_ckpt = "checkpoints/vae_epoch10.pt"
    prior_ckpt = "checkpoints/prior_final.pt"
    if not os.path.exists(vae_ckpt):
        print("VAE checkpoint not found, please train the model first.")
        return
    if not os.path.exists(prior_ckpt):
        print("Prior checkpoint not found, please train the prior model first.")
        return

    subprocess.run(["python", "inference.py", "--config", config, "--vae_ckpt", vae_ckpt, 
                    "--prior_ckpt", prior_ckpt, "--length_seconds", "5"], check=True)

if __name__ == "__main__":
    test_generation()
