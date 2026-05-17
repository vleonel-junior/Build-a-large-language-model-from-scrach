import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import platform
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Fonction pour initialiser un groupe de processus distribué (1 processus / GPU).
# C'est ce qui permet la communication entre les différents processus (les GPU).

def ddp_setup(rank, world_size):
    """
    Arguments:
        rank : ID unique du processus (ex: 0 pour le GPU principal, 1 pour le GPU secondaire, etc.)
        world_size : Le nombre total de processus dans le groupe (ex: 2 si on a 2 GPUs)
    """
    # Pour bien comprendre : dans PyTorch, même si vos GPU sont branchés sur le même ordinateur,
    # le système DDP les traite comme s'ils étaient sur des ordinateurs séparés.
    # Ils vont donc communiquer entre eux via des protocoles réseau (comme Internet).
    
    # Adresse IP de la machine qui exécute le processus principal (rank 0).
    # Comme tous nos GPU sont sur le même ordinateur, on utilise "localhost".
    os.environ["MASTER_ADDR"] = "localhost"
    
    # Un port réseau au choix (doit être libre) pour que les processus communiquent.
    os.environ["MASTER_PORT"] = "12345"

    # Initialisation du groupe de processus
    # C'est ici la "réunion" commence et que le protocole de communication est choisi.
    if platform.system() == "Windows":
        # "libuv" est un module de gestion réseau.
        # Il pose parfois des problèmes avec Windows, donc on le désactive pour éviter des plantages.
        os.environ["USE_LIBUV"] = "0"

        # Sur Windows, NCCL (le backend ultra-rapide de Nvidia) n'est pas souvent pas disponible.
        # On utilise donc "gloo" (créé par Facebook) comme langage de communication de secours.
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # Sur Linux (comme Colab ou les serveurs d'entreprise), on utilise "nccl"
        # NVIDIA Collective Communication Library. C'est le mode le plus rapide et le standrad !
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Assignation du GPU
    # Très important : par défaut, Pytorch a tendance à tout envoyer sur le GPU 0.
    # Ici, on force le processus 0 à utiliser le GPU 0, le processus 1 à utiliser le GPU 1, etc.
    torch.cuda.set_device(rank)


# Définition d'un "Jeu de données jouet" (ToyDataset) personnalisé
# Il hérite de la classe parente 'Dataset' de Pytorch.
class ToyDataset(Dataset):

    # 1. Ici on lui passe nos données (X) et nos cibles (y)
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    # 2. Cette méthode permet de récupérer un élément spécifique par son index
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    # 3. La méthode pour connaître la taille totale du jeu de données.
    def __len__(self):
        return self.labels.shape[0]

# Définition de notre réseau de neurones.
# Il hérite obligatoirement de 'torch.nn.Module', la brique de base de Pytorch.

class NeuralNetwork(torch.nn.Module):
    
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        # nn.Sequential permet de regrouper plusieurs couches en un seul bloc.
        # Les données traverseront ces couches dans l'ordre, de la première à la dernière.
        self.layers = torch.nn.Sequential(
            # 1ère couche cachée
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2ème couche cachée
            torch.nn.Linear(30, 20),    
            torch.nn.ReLU(),

            # Couche de sortie
            torch.nn.Linear(20, num_outputs),
        )

    # La méthode forward (passe avant) définit le trajet des données.
    def forward(self, x):
        logits = self.layers(x)
        return logits

# Fonction pour préparer notre jeu de données et le diviser en un ensemble d'entraînement et de test.

def prepare_dataset():

    # 1. Création de données jouet pour l'entraînement
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    # 2. Création de données pour le test
    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    # Décommentez les lignes suivantes si vous voulez multiplier la taille de ce jeu de données
    # (très utile si vous testez avec beaucoup de GPUs, comme 4 ou 8, pour que chaque GPU ait assez de travail).
    # factor = 4
    # X_train = torch.cat([X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)])
    # y_train = y_train.repeat(factor)
    # X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    # y_test = y_test.repeat(factor)

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    # DataLoader pour l'entrainement :
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        # On met shuffle à False.
        # Pourquoi ? Parce qu'avec DDP, c'est le "DistributedSampled" qui va se charger de 
        # mélanger et distrbuer les données.
        shuffle=False,
        pin_memory=True, # Pour accélérer le transfert RAM vers GPU
        drop_last=True, # Jette le dernier pour éviter les problèmes de batchs incomplets
        # Le DistributedSampled, composante clé du DDP pour les données.
        # Il "découpe" le jeu de données en parts égales pour chaque GPU, sans chevauchement.
        sampler=DistributedSampler(train_ds)
    )

    # DataLoader pour le test :
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False, # Pas besoin de mélanger les données de test vu qu'on les évalue une seule fois
    )

    return train_loader, test_loader

# Avec DDP, chaque GPu va exécuter cette fonction main en parallèle.
def main(rank, world_size, num_epochs):

    # 1. On démarre la communication réseau entre les GPUs
    ddp_setup(rank, world_size)

    # 2. Préparation des données et du modèle
    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)

    # On envoie le modèle sur le GPU assigné à ce processus spécifique
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    # On "enveloppe" notre modèle avec DDP
    # C'est ce qui fait que Pytorch va automatiquement synchroniser
    # les gradients entre avec les autres GPU au moment de l'appel de "loss.backward()"
    model = DDP(model, device_ids=[rank])
    # Note : Le modèle original brut est maintenant caché à l'intérieur et accessible via `model.module`

    # 3. Boucle principale d'entraînement
    for epoch in range(num_epochs):
        # On doit dire au DistributedSampler à quelle epoch nous nous situons.
        # Si on oublie cette ligne, le Sampler va mélanager les données la première fois,
        # puis chaque GPU recevra exactement le même découpage de données à chaque epoch !
        train_loader.sampler.set_epoch(epoch)

        model.train()
        for features, labels in train_loader:

            # On envoie un batch sur le GPU assigné à ce processus
            features, labels = features.to(rank), labels.to(rank)

            logits = model(features)
            loss = F.cross_entropy(logits, labels) # Calcul de l'erreur

            optimizer.zero_grad() # Remise à zéro des gradients précédents
            loss.backward() # Rétropropagation et synchronisation DDP inter-gpus
            optimizer.step() # Mise à jour des poids

            # LOGGING
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")
            
    # 4. Évaluation du modèle
    model.eval()

    try:
        # Chaque GPU calcule son accuracy
        train_acc = compute_accuracy(model, train_loader, device=rank)
        print(f"[GPU{rank}] Training accuracy", train_acc)
        test_acc = compute_accuracy(model, test_loader, device=rank)
        print(f"[GPU{rank}] Test  accuracy", test_acc)

    # Si le dataset est trop petit comparé au nombre de GPUs, un GPUpeut se retrouver
    # avec zéro échantillon à tester (ce qui cause une ZeroDivisionError dans compte accuracy).
    except ZeroDivisionError as e:
        raise ZeroDivisionError(
            f"{e}\n\n Ce script est fait pour 2 GPUs. Exécutez-le comme ceci: \n"
            "CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py\n"
            f"Ou, pour le lancer sur vos {torch.cuda.device_count()} GPUs, décommenter les lignes 121 à 127 (qui augmentent la taille du dataset)."
        )

    # Toujours fermer proprement le groupe de processus pour 
    # vider la mémoire de la carte mère et libérer le port réseau "12345"
    destroy_process_group()


# Fonction pour calculer la précision (accuracy) du modèle
# Elle vérifie le pourcentage de prédictions justes. 
def compute_accuracy(model, dataloader, device):
    # 1. On passe le modèle en mode "évaluation"
    # Cela désactive certains comportements utiles uniquement lors de l'entrainement(comme le Dropount, ...)
    model = model.eval()

    correct = 0.0 # Compteur pour les prédictions correctes
    total_examples = 0 # Compteur pour le nombre total d'exemples évalués

    # 2. On parcourt le dataloader
    for idx, (features, labels) in enumerate(dataloader):
        # On s'assure que les données sont sur le même GPU que le modèle
        features, labels = features.to(device), labels.to(device)

        # 3. On désactive le calcul des gradients pour accélérer l'évaluation
        with torch.no_grad():
            logits = model(features)
        
        # 4. Predictions
        # Les logits sont les scores bruts pour chaque classe.
        # L'index du score le plus élevé est la classe prédicte.
        predictions = torch.argmax(logits, dim=1)

        # On combare les prédictions avec les vraies étiquettes
        # Ça crée un tenseur de 1/0 (Vrai/Faux)
        compare = labels == predictions

        # 5. Mise à jour des compteurs
        # sum(compare) compte le nombre de "True" dans le batch
        correct += torch.sum(compare) 
        total_examples += labels.shape[0] # Ajoute la taille du batch au total

        # On renvoie le pourcentage (bons / total)
        # Le .item() permet d'extraire la valeur numérique du tenseur
        return (correct / total_examples).item()
    

if __name__ == "__main__":
    
    # Ce script risque de planter si vous avez plus de 2 GPUs due à la petite taille du dataset.
    # Si vous avez plus de 2 GPUs, exécutez `CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py`
    
    print("Version de Pytorch:", torch.__version__)
    print("CUDA est-il disponible ? :", torch.cuda.is_available())
    print("Nombre de GPUs disponibles :", torch.cuda.device_count())

    torch.manual_seed(123) # Pour la reproductibilité

    num_epochs = 3

    world_size = torch.cuda.device_count() # Nombre de processus = nombre de GPUs
    # `spawn` injecte AUTOMATIQUEMENT le `rank`(le numéro 0, 1, 2, ...)
    # comme tout premier argument de la fonction `main`
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
    # `nprocs=world_size` signifie on lance exactement 1 processus par GPU disponible.



        




        




