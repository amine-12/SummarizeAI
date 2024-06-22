def download_cnn_dailymail():
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    dataset.save_to_disk('data/cnn_dailymail')

if __name__ == "__main__":
    download_cnn_dailymail()
    print("Datasets downloaded and saved to disk.")