import torch
import torch.nn as nn
import numpy as np
from scipy import sparse
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------
# Device
# ---------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MATRIX_PATH = "data/processed/user_item_matrix.npz"


# ---------------------------------------------------
# Load Sparse Matrix
# ---------------------------------------------------

def load_matrix():
    matrix = sparse.load_npz(MATRIX_PATH)
    print("Matrix shape:", matrix.shape)
    return matrix


# ---------------------------------------------------
# Model
# ---------------------------------------------------

class RecommenderModel(nn.Module):

    def __init__(self, num_users, num_anime, embedding_dim=64):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.anime_embedding = nn.Embedding(num_anime, embedding_dim)

    def forward(self, user_ids, anime_ids):

        user_vec = self.user_embedding(user_ids)
        anime_vec = self.anime_embedding(anime_ids)

        dot = (user_vec * anime_vec).sum(dim=1)

        return torch.sigmoid(dot)


# ---------------------------------------------------
# Dataset
# ---------------------------------------------------

class InteractionDataset(Dataset):

    def __init__(self, matrix):

        self.users, self.items = matrix.nonzero()
        self.num_items = matrix.shape[1]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):

        user = self.users[idx]
        pos_item = self.items[idx]

        if np.random.rand() < 0.5:
            return (
                torch.tensor(user, dtype=torch.long),
                torch.tensor(pos_item, dtype=torch.long),
                torch.tensor(1.0)
            )
        else:
            neg_item = np.random.randint(self.num_items)

            return (
                torch.tensor(user, dtype=torch.long),
                torch.tensor(neg_item, dtype=torch.long),
                torch.tensor(0.0)
            )


# ---------------------------------------------------
# Training
# ---------------------------------------------------

def train_model(matrix):

    num_users, num_items = matrix.shape

    model = RecommenderModel(num_users, num_items).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    dataset = InteractionDataset(matrix)

    loader = DataLoader(
    dataset,
    batch_size=1024,
    shuffle=True,
    num_workers=0
)

    for epoch in range(5):

        total_loss = 0

        for users, items, labels in tqdm(loader):

            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device)

            preds = model(users, items)

            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} Loss {total_loss}")

    return model


# ---------------------------------------------------
# Save Embeddings
# ---------------------------------------------------

def save_embeddings(model):

    user_emb = model.user_embedding.weight.detach().cpu().numpy()
    anime_emb = model.anime_embedding.weight.detach().cpu().numpy()

    np.save("data/processed/user_embeddings.npy", user_emb)
    np.save("data/processed/anime_embeddings.npy", anime_emb)


# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":

    matrix = load_matrix()

    model = train_model(matrix)

    save_embeddings(model)

    print("Training complete.")