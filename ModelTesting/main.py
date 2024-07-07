from utils import *

import os

### FOLDER SELECTION ###
try:
    data_dir, real_dir_path, fake_dir_path = create_gui()
except Exception as e:
    print("An error occurred:", str(e))

# Path normalization to avoid GUI parsing
data_dir = os.path.normpath(data_dir)
real_dir_path = os.path.normpath(real_dir_path)
fake_dir_path = os.path.normpath(fake_dir_path)

### PREPARATION ###
rgb_test_df = create_test_set(os.path.join(real_dir_path, "metadata.csv"), os.path.join(fake_dir_path, "metadata.csv"))

### FFT APPLICATION ###
path_matching_dict = FFT_application(rgb_test_df, real_dir_path, fake_dir_path, os.path.join(data_dir, "fourier"))
test_df = update_testset_paths(rgb_test_df, path_matching_dict)

print(test_df)

### TESTING ###

# Loading the trained model
model = APN_Model()
model.efficientnet.conv_stem = nn.Conv2d(1, 32, 3, 2, 1, bias=False)
model.to(DEVICE)
model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.getcwd()), "best_model.pt")))

# Unzipping the embeddings database
database = unzip_csv(os.path.join(os.path.dirname(os.getcwd()), "embeddings_database.zip"), data_dir)

# Performance evaluation
test_model(data_dir, test_df, model, database)