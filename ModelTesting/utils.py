import tkinter as tk
from tkinter import filedialog

import pandas as pd

import os
import numpy as np
import cv2
from tqdm.notebook import tqdm

import torch
from torch import nn
import timm
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import io
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.backends.backend_pdf import PdfPages
import webbrowser

## GUI CREATION ##

def select_folder(entry):
    """Opens a dialog to select a folder and updates the entry with the selected folder's path."""
    folder_path = filedialog.askdirectory()  # Open the folder selection dialog
    if folder_path:
        entry.delete(0, tk.END)  # Clear the current entry
        entry.insert(0, folder_path)  # Insert the new path

def create_gui():
    """Creates the GUI for folder selection and returns the selected folder paths."""
    root = tk.Tk()
    root.title("Folder Selection GUI")

    # Static message label with user instructions
    message_label = tk.Label(root, text="Please select where generated data will be stored.", fg="blue")
    message_label.grid(row=0, column=1, padx=10, pady=10, sticky="w")

    # Setup entries to display the folder paths
    entry1 = tk.Entry(root, width=50)
    entry1.grid(row=1, column=1, padx=10, pady=10)

    # Static message label with user instructions
    message_label = tk.Label(root, text="Please select folders containing testing images.", fg="blue")
    message_label.grid(row=2, column=1, padx=10, pady=10, sticky="w")

    entry2 = tk.Entry(root, width=50)
    entry2.grid(row=3, column=1, padx=10, pady=10)

    entry3 = tk.Entry(root, width=50)
    entry3.grid(row=4, column=1, padx=10, pady=10)

    # Static message label with user instructions
    message_label = tk.Label(root, text="Make sure to select Root Folders, not inner ones.", fg="red")
    message_label.grid(row=5, column=1, padx=10, pady=10, sticky="w")

    # Buttons to trigger folder selection
    button1 = tk.Button(root, text="Select Storage Folder", command=lambda: select_folder(entry1))
    button1.grid(row=1, column=0, padx=10, pady=10)

    button2 = tk.Button(root, text="Select 'Real' Folder", command=lambda: select_folder(entry2))
    button2.grid(row=3, column=0, padx=10, pady=10)

    button3 = tk.Button(root, text="Select 'Fake' Folder", command=lambda: select_folder(entry3))
    button3.grid(row=4, column=0, padx=10, pady=10)

    # Use a variable to store paths before closing the GUI
    paths = []

    # Button to close the GUI and save the paths
    def close_and_save():
        paths.append(entry1.get())  # Save first path
        paths.append(entry2.get())  # Save second path
        paths.append(entry3.get())  # Save third path
        root.destroy()  # Close the window

    close_button = tk.Button(root, text="Done", command=close_and_save)
    close_button.grid(row=6, column=1, padx=10, pady=10)

    # Start the GUI
    root.mainloop()

    # Return the paths after the window is closed
    return paths[0], paths[1], paths[2]


## PREPARATION ##

def create_test_set(real_csv_path, fake_csv_path):
    """
    Creates a test set dataframe with columns 'real' and 'fake' from two different CSV files.
    Paths are normalized to ensure compatibility with the operating system.

    Parameters:
    - real_csv_path (str): The file path to the CSV containing real image paths.
    - fake_csv_path (str): The file path to the CSV containing fake image paths.

    Returns:
    - test_df (pd.DataFrame): A DataFrame with two columns 'real' and 'fake' containing normalized paths.
    """
    # Load data from CSV files
    real_df = pd.read_csv(real_csv_path)
    fake_df = pd.read_csv(fake_csv_path)

    # Normalize paths in the dataframes
    real_df['image_path'] = real_df['image_path'].apply(os.path.normpath)
    fake_df['image_path'] = fake_df['image_path'].apply(os.path.normpath)

    # Get the smaller length of the two dataframes
    min_length = min(len(real_df), len(fake_df))

    # Truncate both dataframes to the minimum length
    real_df = real_df.head(min_length)
    fake_df = fake_df.head(min_length)

    # Create a new DataFrame combining the two
    test_df = pd.DataFrame({
        'real': real_df['image_path'],
        'fake': fake_df['image_path']
    })

    return test_df

## FFT APPLICATION ##

def greyscale_FFT(img_path):
	"""
	Applies Fast Fourier Transform (FFT) to a greyscale image and returns its magnitude spectrum.

	Parameters
	----------
		img_path (str): input image path.

	Returns
	-------
		fft_img (np.ndarray): a 2D array representing the magnitude spectrum of the FFT of the input image, normalized to the range [0, 255].
	"""
	
	# Read the image from the specified path in BGR color format
	RGBimg = cv2.imread(img_path)
	
	# Convert the image from BGR to grayscale
	grayImg = cv2.cvtColor(RGBimg, cv2.COLOR_BGR2GRAY)
	
	# Apply the 2D FFT to the grayscale image
	fft_img = np.fft.fft2(grayImg)
	
	# Compute the logarithm of the absolute value of the FFT to get the magnitude
	fft_img = np.log(np.abs(fft_img))

	# Find the minimum and maximum values of the magnitude for normalization
	min_val = np.min(fft_img)
	max_val = np.max(fft_img)
	
	# Normalize the magnitude image to the range [0, 255]
	fft_img = (fft_img - min_val) * (255.0 / (max_val - min_val))
	
	# Convert the normalized image to uint8 (integer values from 0 to 255)
	fft_img = np.uint8(fft_img)

	# Return the normalized magnitude image
	return fft_img

def FFT_application(df, real_dir_path, fake_dir_path, output_dir):
    """
    Applies FFT to images listed in the dataframe and saves the results in a specified directory,
    handling relative paths by prefixing them with the appropriate directory paths.

    Parameters
    ----------
    df (pd.DataFrame): DataFrame containing the image paths under 'real' and 'fake' columns.
    real_dir_path (str): The directory path where real images are stored.
    fake_dir_path (str): The directory path where fake images are stored.
    output_dir (str): The directory where the FFT images will be saved.
    
    Returns
    -------
    path_mapping_dict (dict): Dictionary mapping original image paths to their new FFT-transformed image paths.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    path_mapping_dict = {}

    # Iterate over all images in the dataframe under both 'real' and 'fake' columns
    for column in ['real', 'fake']:
        for img_path in tqdm(df[column], desc=f"Applying FFT on {column} images"):
            # Determine full image path based on column
            full_img_path = os.path.join(real_dir_path if column == 'real' else fake_dir_path, img_path)

            # Apply the greyscale FFT
            fft_img = greyscale_FFT(full_img_path)

            # Generate the new image path
            new_img_name = img_path.replace(os.sep, "+").replace("img", f"{column}")
            new_img_path = os.path.join(output_dir, new_img_name)

            # Save the transformed image
            cv2.imwrite(new_img_path, fft_img)
            
            # Update the dictionary
            path_mapping_dict[img_path] = os.path.join("fourier", new_img_name)
    
    return path_mapping_dict

def update_testset_paths(df, path_mapping_dict):
    """
    Updates the paths in the DataFrame columns 'real' and 'fake' using a provided mapping dictionary.

    Parameters
    ----------
    df (pd.DataFrame): DataFrame containing the columns 'real' and 'fake' with image paths to be updated.
    path_mapping_dict (dict): Dictionary mapping original image paths to their new FFT-transformed image paths.

    Returns
    -------
    df (pd.DataFrame): DataFrame with updated image paths in the 'real' and 'fake' columns.
    """
    # Apply the mapping to the 'real' and 'fake' columns
    df['real'] = df['real'].map(path_mapping_dict).fillna(df['real'])
    df['fake'] = df['fake'].map(path_mapping_dict).fillna(df['fake'])

    return df

## TESTING ##

# Device to run calculations on 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class APN_Model(nn.Module):
    """
    Defines a neural network model class APN_Model that uses an EfficientNet (specifically the B0 version) as its backbone.
    """

    def __init__(self, emb_size = 512):
        """
        Initializes the APN_Model with a specific model and a classifier that outputs embedding vector of the specified size.

        Parameters
        ----------
        emb_size (int, optional): the size of the output embedding vector (default is 512).
        """
        super(APN_Model, self).__init__()

        # Define the model to use 
        self.efficientnet = timm.create_model('tf_efficientnetv2_b0', pretrained = False)
        
        # Replace the classifier layer with a linear layer that outputs embeddings of size `emb_size`
        self.efficientnet.classifier = nn.Linear(in_features=self.efficientnet.classifier.in_features, out_features = emb_size)

    def forward(self, images):
        """
        Performs the forward pass of the model, which takes a batch of images and returns their embeddings.

        Parameters
        ----------
            images (torch.Tensor): a batch of images to process.

        Returns
        -------
            embeddings (torch.Tensor): a batch of embeddings of size `emb_size`.
        """
        embeddings = self.efficientnet(images)
        return embeddings

def unzip_csv(zip_file_name, data_dir):
    """
    Extract a CSV file from a ZIP archive and loads it into a pandas DataFrame.

    Parameters
    ----------
        zip_file_name (str): The name of the ZIP file to extract.
        data_dir (str): path to the Storage folder.

    Returns
    -------
        database (pd.DataFrame): The DataFrame containing the extracted CSV file.
    """
    extraction_path = os.path.join(data_dir, "embeddings_database")
    
    # Open the ZIP file
    with zipfile.ZipFile(zip_file_name, 'r') as zipf:
        # Extract all the contents into the current directory
        zipf.extractall(extraction_path)

    # Read the extracted file as a DataFrame
    database = pd.read_csv(os.path.join(extraction_path, "database.csv"))

    return database

def getImageEmbeddings(img, model):
    """
    Generates embeddings for a given image using the provided model.

    Parameters
    ----------
        img (numpy.ndarray): the input image as a NumPy array.
        model (torch.nn.Module): the PyTorch model used to generate the image embeddings.

    Returns
    -------
        img_enc (numpy.ndarray): the embeddings of the input image.
    """
    # Add a new dimension to the image array to match the expected input shape of the model
    img = np.expand_dims(img, 0)
    
    # Convert the NumPy array to a PyTorch tensor and normalize pixel values to the range [0, 1]
    img = torch.from_numpy(img) / 255.0
    
    # Set the model to evaluation mode
    model.eval()
    
    # Disable gradient calculation for efficiency
    with torch.no_grad():
        # Move the image tensor to the appropriate device (CPU or GPU)
        img = img.to(DEVICE)
        
        # Add a batch dimension, pass the image through the model to get the embeddings
        img_enc = model(img.unsqueeze(0))
        
        # Detach the embeddings from the computation graph and move them back to the CPU
        img_enc = img_enc.detach().cpu().numpy()
        
        # Convert the embeddings to a NumPy array
        img_enc = np.array(img_enc)

    return img_enc

def euclidean_dist(img_enc, anc_enc_arr):
    """
    Computes the Euclidean distance between a given image encoding and an array of anchor encodings.

    Parameters
    ----------
        img_enc (numpy.ndarray): a 1D array representing the encoding of the image.
        anc_enc_arr (numpy.ndarray): a 2D array representing the anchor images and their encodings.

    Returns
    -------
        dist (numpy.ndarray): a 1D array containing the Euclidean distances between the image encoding and each of the anchors' one.
    """
    #dist = np.sqrt(np.dot(img_enc-anc_enc_arr, (img_enc- anc_enc_arr).T))
    dist = np.dot(img_enc-anc_enc_arr, (img_enc- anc_enc_arr).T)
    #dist = np.sqrt(dist)
    
    return dist

def searchInDatabase(img_enc, database):
    """
    Searches for the closest match to a given image embedding in a database of embeddings.

    Parameters
    ----------
        img_enc (numpy.ndarray): the embedding of the input image.
        database (pandas.DataFrame): a DataFrame containing image embeddings with 'anchor' column for image names and subsequent columns for embedding values.

    Returns
    -------
        (str): the name of the anchor image in the database that is closest to the input image embedding.
    """
    # Extract the embeddings and image names from the database
    anc_enc_arr = database.iloc[:, 1:].to_numpy()  # All columns except the first one (assuming embeddings start from the second column)

    # Initialize an empty list to store distances
    distance = []

    # Iterate over each embedding in the database
    for i in range(anc_enc_arr.shape[0]):
        # Compute the Euclidean distance between the input embedding and the current database embedding
        dist = euclidean_dist(img_enc, anc_enc_arr[i : i+1, :])
        # Append the computed distance to the distance list
        distance = np.append(distance, dist)

    # Find the index of the closest embedding by sorting distances
    closest_idx = np.argsort(distance)

    # Return the name of the closest anchor image
    return database['anchor'][closest_idx[0]]

def prediction_fn(data_dir, test_df, currentTest, y_pred, model, database):
    """
    Tests model predictive capability.

    Parameters
    ----------
        data_dir (str): path to the directory containing FFT images.
        test_df (pandas.DataFrame): DataFrame containing the test images.
        currentTest (str): desired images to test the model on.
        y_pred (list): a list to store the predictions ('real' or 'fake') for each test image.
        model (torch.nn.Module): the PyTorch model used to generate the image embeddings.
        database (pandas.DataFrame): the DataFrame containing image embeddings with 'Anchor' column for image names and subsequent columns for embedding values.
    """

    # Iterate over each row in the test DataFrame with progress bar
    for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # Construct the full image path
        img_name = os.path.join(data_dir, row[currentTest])
        
        # Read the image from the path
        img = io.imread(img_name)

        # Get the image embeddings using the model
        img_enc = getImageEmbeddings(img, model)

        # Search for the closest match in the database
        closestLabel = searchInDatabase(img_enc, database)

        # Determine if the closest match is 'real' or 'fake' and append the result to pred_vector
        if "real" in closestLabel:
            y_pred.append("real")
        else:
            y_pred.append("fake")

def calculate_metrics(test_df, y_pred, y_true, output_path='metrics_report.pdf'):
    """
    Calculate classification metrics, display and save them in a PDF file along with the confusion matrix.

    Parameters
    ----------
        test_df (pandas.DataFrame): DataFrame containing the test data.
        y_pred (array-like): Predicted labels for the test set.
        y_true (array-like): True labels for the test set; overridden internally based on the `test_df`.
        output_path (str): Path where the PDF file will be saved.
    """
    # Create ground truth vectors
    y_true = np.array(['real'] * len(test_df) + ['fake'] * len(test_df))

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=["real", "fake"])
    TN, FP, FN, TP = cm.ravel()

    # Metrics calculation
    accuracy = round((TP + TN) / (TP + TN + FP + FN), 4) * 100
    precision = round(TP / (TP + FP), 4) * 100
    recall = round(TP / (TP + FN), 4) * 100
    specificity = round(TN / (TN + FP), 4) * 100
    F1_score = round(2 * (precision * recall) / (precision + recall), 2)

    metrics_dict = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": F1_score
    }

    # Check if the file exists and warn about overwriting
    if os.path.exists(output_path):
        response = input(f"The file {output_path} already exists. Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return

    # Prepare to save to PDF
    with PdfPages(output_path) as pdf:
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        # Create a text report of classification report
        plt.figure(figsize=(8, 6))
        cl_report = classification_report(y_true, y_pred, output_dict=True)
        sns.heatmap(pd.DataFrame(cl_report).iloc[:-1, :].T, annot=True, fmt=".2f", cmap='Blues')
        plt.title('Classification Report')
        pdf.savefig()
        plt.close()

        # Print and save metrics
        plt.figure(figsize=(8, 6))
        for metric, value in metrics_dict.items():
            print(f"{metric}: {value}")
        plt.bar(metrics_dict.keys(), metrics_dict.values(), color='blue')
        plt.title('Performance Metrics')
        plt.ylabel('Percentage')
        plt.ylim(0, 100)
        pdf.savefig()
        plt.close()

def test_model(data_dir, test_df, model, database):
    """
    Tests a given model on real and fake images, collects predictions, and calculates performance metrics.
    
    This function orchestrates the testing process by invoking predictions on both real and fake images
    using a provided model and database. It collects the predictions, compares them with true labels, and
    computes the classification metrics, which are then saved to a PDF report.

    Parameters
    ----------
        data_dir (str): path to the directory containing FFT images.
        test_df (pandas.DataFrame): DataFrame containing the test images.
        model (torch.nn.Module): the PyTorch model used to generate the image embeddings.
        database (pandas.DataFrame): the DataFrame containing image embeddings with 'Anchor' column for image names and subsequent columns for embedding values.

    Returns
    -------
    None
        This function does not return any value but will generate a PDF report at the specified location
        within `data_dir` containing the performance metrics of the model.
    """
    y_true = []  # This will hold the true labels, assumed to be prepared or set within `calculate_metrics`.
    y_pred = []  # List to collect predictions from both real and fake image tests.

    # Testing on REAL images: Collect predictions using a specified function
    prediction_fn(data_dir, test_df, 'real', y_pred, model, database)

    # Testing on FAKE images: Similarly, collect predictions for fake images
    prediction_fn(data_dir, test_df, 'fake', y_pred, model, database)

    # Calculate metrics and generate a report: The function calculates various classification metrics
    # and saves them to a PDF file in the specified data directory.
    calculate_metrics(test_df, y_pred, y_true, os.path.join(data_dir, "test_report.pdf"))

    # Automatically open the generated PDF report with the default PDF viewer
    webbrowser.open('file://' + os.path.realpath(os.path.join(data_dir, "test_report.pdf")))

