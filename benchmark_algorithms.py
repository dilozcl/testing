from pprint import pprint
import os
import pandas as pd
import plotly.express as px
import torch
from IPython.display import Image
from torchradio import DeviceLogs, Receiver, Transmitter
from torchradio.algorithm import DSSS, Modem
from torchradio.env.null import ControlledSNREnvironment
import matplotlib.pyplot as plt


modem_dict = {
    "BPSK": Modem("psk", 2),
    "QPSK": Modem("psk", 4),
    "PSK64": Modem("psk", 64),
    "QAM16": Modem("qam", 16),
    "DSSS-4-BPSK": DSSS(torch.randint(0, 2, (4,))),
    "DSSS-8-BPSK": DSSS(torch.randint(0, 2, (8,))),
}

test_algorithms = {
    modem_name: {"tx": Transmitter(modem.tx), "rx": Receiver(modem.rx)}
    for modem_name, modem in modem_dict.items()
}

transmitters, receivers = (
    {
        algorithm_name: algorithm[x]
        for algorithm_name, algorithm in test_algorithms.items()
    }
    for x in ["tx", "rx"]
)

env = ControlledSNREnvironment(0)

transmitter_name = "tx-test"
receiver_name = "rx-test"
algorithm = "QPSK"
env.place(
    {transmitter_name: transmitters[algorithm]},
    {receiver_name: receivers[algorithm]},
)
pprint(env.devices)



env.place(
    {transmitter_name: transmitters[algorithm]},
    {receiver_name: receivers[algorithm]},
)
device_logs = env.simulate(200)



def _analyze(device_logs: DeviceLogs, *, verbose: bool = False) -> dict[str, float]:
    # get transmitter and receiver names
    transmitter_names = list(device_logs.tx.keys())
    receiver_names = list(device_logs.rx.keys())

    # check device_logs only contain a single tx/rx pair
    assert len(transmitter_names) == 1
    assert len(receiver_names) == 1

    transmitter_name = transmitter_names[0]
    receiver_name = receiver_names[0]

    # transmitted and received bits
    original_bits = device_logs.tx[transmitter_name].metadata["bits"]
    recovered_bits = device_logs.rx[receiver_name]["bits"]
    matched_bits = recovered_bits == original_bits
    bit_error_rate = 1 - torch.mean(matched_bits.float()).item()

    # separate received signal and noise
    background_noise = device_logs.rx[receiver_name]["noise"]
    rx_pure_signal = device_logs.rx[receiver_name]["raw"] - background_noise
    snr = (
        10 * torch.log10(torch.var(rx_pure_signal) / torch.var(background_noise)).item()
    )

    # throughput
    n_bits = original_bits.shape[-1]
    signal_length = device_logs.tx[transmitter_name].signal.shape[-1]
    throughput = n_bits / signal_length

    if verbose:
        print(f"Basic Analysis for {transmitter_name} to {receiver_name}:")
        print(f"- Bit Error Rate: {100 * bit_error_rate:.2f}%")
        print(f"- SNR: {snr:.2f}dB")
        print(f"- Throughput: {throughput} bits per sample")

    return {"Bit Error Rate": bit_error_rate, "SNR (dB)": snr, "Throughput": throughput}


analysis = _analyze(device_logs, verbose=True)


results_dict = {"Algorithm": []}
n_timesteps = 5120
for snr in torch.linspace(-15, 5, 40):
    env.set_snr(snr)
    for algorithm_name, devices in test_algorithms.items():
        env.place(
            {f"{algorithm_name}-tx": devices["tx"]},
            {f"{algorithm_name}-rx": devices["rx"]},
        )
        device_logs = env.simulate(n_timesteps)
        result = _analyze(device_logs, verbose=False)

        results_dict["Algorithm"].append(algorithm_name)

        for k, v in result.items():
            if k not in results_dict:
                results_dict[k] = []
            results_dict[k].append(v)


results_df = pd.DataFrame(results_dict)
results_df.head()

fig, axs = plt.subplots(figsize=(12, 4))

results_df.plot(ax=axs)



fig = px.line(
    results_df,
    x="SNR (dB)",
    y="Bit Error Rate",
    color="Algorithm",
    title="Bit Error Rate vs SNR",
    log_y=True,
)
Image(fig.to_image(format="png"))


throughput_df = results_df.groupby("Algorithm")["Throughput"].mean().reset_index()
fig = px.bar(
    throughput_df,
    x="Algorithm",
    y="Throughput",
    log_y=False,
    title="Bit Error Rate vs SNR",
)
# Image(fig.to_image(format="png"))
fig.write_image(os.path("/home/diego_lozano/","BER_vs_SNR.png"))




def _normalize_column(df: pd.DataFrame, column_name: str) -> None:
    df[f"Normalized {column_name}"] = df[column_name] / df[column_name].abs().max()


# isolate low SNR results
snr_limit = -14.9
low_snr_results = results_df[results_df["SNR (dB)"] < snr_limit].copy()

# "explode" each row with tradeoff parameter alpha
low_snr_results["alpha"] = [
    torch.linspace(0, 1, 20).numpy().tolist() for _ in range(len(low_snr_results.index))
]
low_snr_results = low_snr_results.explode("alpha").reset_index()

# normalize columns to make scales for tradeoff metrics similar
_normalize_column(low_snr_results, "Throughput")
_normalize_column(low_snr_results, "Bit Error Rate")

# compute scores
low_snr_results["Score"] = low_snr_results.apply(
    lambda x: (1 - x["alpha"]) * (1 / x["Normalized Bit Error Rate"])
    + x["alpha"] * x["Normalized Throughput"],
    axis=1,
)
_normalize_column(low_snr_results, "Score")

fig = px.line(
    low_snr_results,
    x="alpha",
    y="Normalized Score",
    color="Algorithm",
    title="Tradeoff Curve for Bit Error Rate at -10dB vs. Throughput",
    labels={"alpha": "Throughput Priority"},
)
Image(fig.to_image(format="png"))

