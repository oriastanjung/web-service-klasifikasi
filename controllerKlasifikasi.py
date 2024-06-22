import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from helper.ekstraksiBentuk import ekstrakBentuk
from helper.ekstraksiTekstur import ekstrakTekstur
# import pickle
from scipy.spatial.distance import cdist
from numpy.linalg import pinv

labels = ['Avicennia alba', 'Bruguiera cylindrica', 'Bruguiera gymnorrhiza','Lumnitzora littorea', 'Rhizophora apiculata', 'Rhizophora mucronata','Scyphyphora hydrophyllacea', 'Senoratia alba', 'Xylocarpus granatum']

# Load data from CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Preprocess data
def preprocess_data(df):
    # Extract features and labels
    X = df[['Panjang','Keliling','Diameter','Luas','Faktor bentuk',
            'ASM 0','ASM 45','ASM 90','ASM 135','Kontras 0','Kontras 45',
            'Kontras 90','Kontras 135','IDM 0','IDM 45','IDM 90','IDM 135','Entropy 0','Entropy 45','Entropy 90',
            'Entropy 135','Korelasi 0','Korelasi 45',
            'Korelasi 90','Korelasi 135']].values
    y = df['Jenis'].values

# Standardize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y, scaler, label_encoder  # Return scaler along with X, y, and label_encoder

# Define the RBFN class for multi-class classification
class RBFNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.centers = None
        self.width = None
        self.weights = None

    def _gaussian(self, X, centers, width):
        return np.exp(-cdist(X, centers) ** 2 / (2 * (width ** 2)))

    def _one_hot_encoding(self, y):
        one_hot = np.zeros((y.shape[0], self.output_size))
        for i, val in enumerate(y):
            one_hot[i, val] = 1
        return one_hot

    def fit(self, X, y):
        # Initialize centers using K-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.hidden_size, random_state=42)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        # Calculate width
        self.width = np.mean(cdist(self.centers, self.centers)) / np.sqrt(2 * self.hidden_size)

        # Compute activations
        phi = self._gaussian(X, self.centers, self.width)

        # Solve for weights using Moore-Penrose pseudoinverse
        phi_pseudo_inverse = pinv(phi)
        one_hot_y = self._one_hot_encoding(y)
        self.weights = np.dot(phi_pseudo_inverse, one_hot_y)

    def predict(self, X):
        phi = self._gaussian(X, self.centers, self.width)
        predictions = np.dot(phi, self.weights)
        return np.argmax(predictions, axis=1)

    def save_model(self, file_path):
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump((self.centers, self.width, self.weights), f)

    @staticmethod
    def load_model(file_path):
        import pickle
        with open(file_path, 'rb') as f:
            centers, width, weights = pickle.load(f)
        model = RBFNN(input_size=centers.shape[1], hidden_size=centers.shape[0], output_size=weights.shape[1])
        model.centers = centers
        model.width = width
        model.weights = weights
        return model

# Load and preprocess training data
train_file_path = "data_set.csv"
train_df = load_data(train_file_path)
X_train, y_train, scaler, label_encoder = preprocess_data(train_df)

# Load the model from the file
loaded_model = RBFNN.load_model('RBFNN_model.pkl')


def klasifikasiMangrove(image) : 
    major_axis_length,perimeter,diameter,area,shape_factor=ekstrakBentuk(image)
    asm_result,kontras_result,idm_result,entropy_result,korelasi_result=ekstrakTekstur(image)

    
    new_data = np.array([[major_axis_length,perimeter,diameter,area,shape_factor,asm_result[0],asm_result[1],asm_result[2],asm_result[3],kontras_result[0],kontras_result[1],kontras_result[2],kontras_result[3],idm_result[0],idm_result[1],idm_result[2],idm_result[3],entropy_result[0],entropy_result[1],entropy_result[2],entropy_result[3],korelasi_result[0],korelasi_result[1],korelasi_result[2],korelasi_result[3]]])  # Example
    new_data_scaled = scaler.transform(new_data)
    predicted_class = loaded_model.predict(new_data_scaled)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    
    
    klasifikasi_mangrove =labels[predicted_label[0]-1]
    id_mangrove = predicted_label[0]-1
    print(major_axis_length,perimeter,diameter,area,shape_factor)

    return klasifikasi_mangrove, id_mangrove
