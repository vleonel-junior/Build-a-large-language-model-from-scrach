## Chapitre 3 — Mécanismes d'attention

### Introduction

Ce chapitre porte sur les **mécanismes d'attention**, qui constituent le cœur de l'architecture des LLMs. L'objectif est d'en comprendre le fonctionnement de manière isolée et mécanique, avant de les intégrer dans le modèle complet au chapitre 4.

<div align="center">

![Figure 3.1](img/figure_3.1.png)

*Figure 3.1 : Les trois grandes étapes de la construction d'un LLM. Ce chapitre se concentre sur l'étape 2 de la phase 1 : l'implémentation des mécanismes d'attention.*

</div>

Quatre variantes du mécanisme d'attention seront implémentées progressivement, chacune construisant sur la précédente :

1. **Self-attention simplifiée** — version épurée, sans poids apprenables, pour saisir la logique fondamentale.
2. **Self-attention avec poids apprenables** — version complète, entraînable.
3. **Attention causale** (*causal attention*) — un masque est ajouté pour que le modèle ne puisse pas « voir » les tokens futurs lors de la génération, token par token.
4. **Attention multi-têtes** (*multi-head attention*) — plusieurs mécanismes d'attention opèrent en parallèle, permettant au modèle de capturer simultanément différents aspects des relations entre tokens.

<div align="center">

![Figure 3.2](img/figure_3.2.png)

*Figure 3.2 : Les quatre variantes d'attention implémentées dans ce chapitre, de la self-attention simplifiée jusqu'à l'attention multi-têtes.*

</div>

L'implémentation finale de l'attention multi-têtes sera directement réutilisée dans l'architecture du LLM au chapitre suivant.

---

### 3.1 Le problème de la modélisation des longues séquences

Avant d'introduire le mécanisme de *self-attention*, il est utile de comprendre le problème qu'il vient résoudre — et pourquoi les architectures qui le précèdent étaient insuffisantes.

#### Le contexte : la traduction automatique

Prenons l'exemple d'un modèle de traduction. Une traduction mot à mot est impossible : les structures grammaticales des langues source et cible divergent trop profondément.

<div align="center">

![Figure 3.3](img/figure_3.3.png)

*Figure 3.3 : Traduire mot à mot est insuffisant. La traduction exige une compréhension du contexte global et un réalignement grammatical entre les deux langues.*

</div>

La solution classique consiste à utiliser une architecture **encodeur–décodeur** : l'encodeur lit et compresse la séquence d'entrée, le décodeur produit la séquence traduite à partir de cette représentation compressée.

#### L'architecture encodeur–décodeur avec RNN

Avant les Transformers, les **réseaux de neurones récurrents** (RNNs) dominaient cette tâche. Un RNN traite la séquence token par token, en maintenant un **état caché** (*hidden state*) mis à jour à chaque étape — une sorte de mémoire interne qui se propage au fil de la séquence.

Dans un RNN encodeur–décodeur :
- L'**encodeur** parcourt toute la séquence d'entrée et tente de condenser son sens dans un **unique vecteur d'état caché final**.
- Le **décodeur** prend ce vecteur comme point de départ et génère la traduction token par token, en maintenant son propre état caché à chaque étape.

<div align="center">

![Figure 3.4](img/figure_3.4.png)

*Figure 3.4 : Dans un RNN encodeur–décodeur, l'encodeur compresse toute la séquence source en un unique état caché final, que le décodeur utilise ensuite pour générer la traduction token par token.*

</div>

#### La limite fondamentale

Ce vecteur d'état caché final constitue le **goulot d'étranglement** de l'architecture. Toute l'information de la séquence d'entrée — quelle que soit sa longueur — doit tenir dans ce seul vecteur. Lors du décodage, le modèle n'a plus accès aux états cachés intermédiaires de l'encodeur : il ne dispose que de cette représentation compressée.

Sur des phrases courtes, cela fonctionne. Sur des séquences longues avec des dépendances distantes, le contexte se dilue et se perd — c'est la **perte de contexte à longue portée**.

> C'est précisément cette limitation qui a motivé l'invention des mécanismes d'attention : plutôt que de forcer toute l'information dans un unique vecteur, permettre au décodeur d'**accéder directement à tous les états cachés de l'encodeur**, en pondérant leur pertinence selon le token en cours de génération.

---

### 3.2 Capturer les dépendances avec les mécanismes d'attention

#### L'attention de Bahdanau (2014) — première rupture

Pour contourner le goulot d'étranglement du vecteur caché unique, Bahdanau et al. proposent en 2014 une modification du RNN encodeur–décodeur : plutôt que de transmettre uniquement l'état caché final, **tous les états cachés intermédiaires de l'encodeur restent accessibles**. À chaque étape de décodage, le décodeur calcule un score de pertinence pour chacun d'eux et pondère sa lecture en conséquence — c'est ce qu'on appelle les **poids d'attention** (*attention weights*).

<div align="center">

![Figure 3.5](img/figure_3.5.png)

*Figure 3.5 : Avec un mécanisme d'attention, le décodeur peut accéder sélectivement à tous les tokens d'entrée. Certains tokens sont plus pertinents que d'autres pour générer un token donné en sortie — cette pertinence est quantifiée par les poids d'attention.*

</div>

#### Le Transformer (2017) — deuxième rupture

Trois ans plus tard, une découverte plus radicale : **le RNN lui-même n'est pas nécessaire**. L'architecture Transformer, proposée en 2017, conserve le principe des poids d'attention mais abandonne complètement la récursivité. Elle introduit la ***self-attention*** : un mécanisme par lequel chaque token d'une séquence peut directement peser la pertinence de **tous les autres tokens de la même séquence**, sans passer par des états cachés intermédiaires.

#### La self-attention en une phrase

> Pour calculer sa représentation, chaque token consulte l'ensemble de la séquence et décide, via des poids appris, à quels autres tokens il doit « prêter attention ».

C'est ce mécanisme qui est au cœur des LLMs modernes comme GPT, et c'est ce que ce chapitre va implémenter de zéro.

<div align="center">

![Figure 3.6](img/figure_3.6.png)

*Figure 3.6 : La self-attention permet à chaque position de la séquence d'interagir avec toutes les autres et d'en pondérer l'importance. Ce chapitre en code l'implémentation complète.*

</div>

---
### 3.3 La self-attention : porter attention à différentes parties de l'entrée

#### Le « self » de self-attention

Le terme *self* désigne le fait que le mécanisme opère **au sein d'une seule et même séquence** : chaque token calcule ses poids d'attention en se comparant à tous les autres tokens de la même séquence. C'est ce qui le distingue de l'attention classique de Bahdanau, où les poids sont calculés entre deux séquences distinctes (entrée et sortie).

---

### 3.3.1 Self-attention simplifiée — sans poids apprenables

Avant d'introduire les poids apprenables, on implémente une version épurée pour saisir la logique fondamentale. L'objectif est de calculer, pour chaque token d'entrée $x^{(i)}$, un **vecteur de contexte** $z^{(i)}$ — une version enrichie de son embedding, qui intègre l'information de tous les autres tokens de la séquence.

<div align="center">

![Figure 3.7](img/figure_3.7.png)

*Figure 3.7 : Pour chaque token d'entrée $x^{(i)}$, la self-attention calcule un vecteur de contexte $z^{(i)}$ en combinant tous les vecteurs d'entrée, pondérés par les poids d'attention $\alpha_{21}$ à $\alpha_{2T}$. Ici, on illustre le calcul de $z^{(2)}$ à partir du token « journey ».*

</div>

On travaille sur la phrase *"Your journey starts with one step."*, déjà encodée en vecteurs de dimension 3 :

```python
import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]]  # step     (x^6)
)
```

Le calcul du vecteur de contexte $z^{(2)}$ se déroule en trois étapes.

---

#### Étape 1 — Calculer les scores d'attention (produit scalaire)

Pour le token *query* $x^{(2)}$ (« journey »), on calcule son **produit scalaire** (*dot product*) avec chacun des tokens de la séquence. Ce score mesure la similarité : plus il est élevé, plus les deux tokens sont alignés dans l'espace vectoriel et plus l'un doit « prêter attention » à l'autre.

```python
query = inputs[1]  # x^(2) : "journey"

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

print(attn_scores_2)
```

```
tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
```

<div align="center">

![Figure 3.8](img/figure_3.8.png)

*Figure 3.8 : Les scores d'attention $\omega_{21}$ à $\omega_{2T}$ sont calculés comme le
produit scalaire entre le vecteur requête $x^{(2)}$ et chaque vecteur d'entrée.*

</div>

> **Produit scalaire.** C'est une multiplication terme à terme de deux vecteurs, dont on somme les résultats. Il est équivalent d'écrire la boucle explicite ou d'utiliser
> `torch.dot` :
> ```python
> res = 0.
> for idx, element in enumerate(inputs[0]):
>     res += inputs[0][idx] * query[idx]
> print(res)                          # tensor(0.9544)
> print(torch.dot(inputs[0], query))  # tensor(0.9544)
> ```

---

#### Étape 2 — Normaliser pour obtenir les poids d'attention (softmax)

On normalise les scores pour obtenir des **poids d'attention** qui somment à 1, les rendant interprétables comme des importances relatives.

Une normalisation naïve par la somme produit bien des poids valides :

```python
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())
```

```
Attention weights: tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
Sum: tensor(1.0000)
```

En pratique, on lui préfère systématiquement la **softmax** — pour deux raisons profondes.

##### Pourquoi la softmax et pas la normalisation simple ?

**Raison 1 — Stabilité numérique.** La softmax naïve calcule :

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

En float32, $e^x$ produit `inf` dès $x \approx 89$ (*overflow*) et s'annule à zéro dès
$x \approx -104$ (*underflow*). Sur de grandes séquences, les scores d'attention peuvent
facilement franchir ces seuils :

```python
torch.exp(torch.tensor(89.0))    # → tensor(inf) : overflow
torch.exp(torch.tensor(-104.0))  # → tensor(0.)  : underflow
```

On obtient alors `inf/inf = NaN` ou `0/0 = NaN`, et le gradient devient inexploitable.
PyTorch contourne cela avec l'astuce **log-sum-exp** : on soustrait le maximum des scores
avant d'exponentier,

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

ce qui est mathématiquement identique (le $e^{-\max(x)}$ se simplifie en numérateur et
dénominateur) mais numériquement stable : la valeur la plus grande devient $e^0 = 1$,
toutes les autres tombent dans $(0, 1]$.

**Raison 2 — Propriétés de gradient favorables.** Comparons les Jacobiens des deux
normalisations.

*Normalisation simple* $w_i = x_i / S$, $S = \sum_j x_j$ :

$$\frac{\partial w_i}{\partial x_i} = \frac{1 - w_i}{S} \qquad
\frac{\partial w_i}{\partial x_j} = \frac{-w_i}{S} \quad (j \neq i)$$

Le gradient dépend de $S$, la somme brute des scores : si $S$ est grand, le gradient
s'effondre et le signal de mise à jour se propage mal. De plus, cette formule exige des
scores positifs — ce que le produit scalaire ne garantit pas.

*Softmax* $w_i = e^{x_i} / Z$, $Z = \sum_k e^{x_k}$ :

> **Démonstration du Jacobien.** On applique la règle du quotient sur $w_i = e^{x_i}/Z$.
>
> *Cas $i = j$ :*
> $$\frac{\partial w_i}{\partial x_i}
>   = \frac{e^{x_i} Z - e^{x_i} e^{x_i}}{Z^2}
>   = \frac{e^{x_i}}{Z} - \left(\frac{e^{x_i}}{Z}\right)^2
>   = w_i - w_i^2
>   \;\Rightarrow\; \boxed{w_i(1 - w_i)}$$
>
> *Cas $i \neq j$ :* $\frac{\partial e^{x_i}}{\partial x_j} = 0$, mais
> $\frac{\partial Z}{\partial x_j} = e^{x_j}$, donc :
> $$\frac{\partial w_i}{\partial x_j}
>   = \frac{0 \cdot Z - e^{x_i} e^{x_j}}{Z^2}
>   = -\frac{e^{x_i}}{Z}\cdot\frac{e^{x_j}}{Z}
>   \;\Rightarrow\; \boxed{-w_i w_j}$$
>
> Les deux cas s'unifient avec le delta de Kronecker $\delta_{ij}$ :
> $$\boxed{\frac{\partial w_i}{\partial x_j} = w_i(\delta_{ij} - w_j)}$$

Ce Jacobien présente trois avantages concrets :

- **Gradient indépendant de l'échelle brute** : il ne dépend que de $w_i \in (0,1)$. Le signal reste dans un intervalle contrôlé quel que soit l'ordre de grandeur des scores.
- **Gradient strictement non nul** : puisque $w_i \in (0,1)$ strictement, $w_i(1-w_i) > 0$. Le signal de correction ne s'éteint jamais silencieusement.
- **Amplification des différences** : l'exponentielle accentue les écarts entre scores. Un petit avantage $\epsilon$ en score produit un avantage plus marqué en poids, générant des patterns d'attention plus tranchés et des gradients plus informatifs.


##### Implémentation

```python
# Version naïve — instable sur grandes valeurs
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

# Version PyTorch — log-sum-exp intégré, à utiliser en pratique
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())
```

```
Attention weights: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
Sum: tensor(1.)
```

<div align="center">

![Figure 3.9](img/figure_3.9.png)

*Figure 3.9 : Les scores d'attention $\omega_{21}$ à $\omega_{2T}$ sont normalisés via softmax
pour produire les poids d'attention $\alpha_{21}$ à $\alpha_{2T}$, qui somment à 1.*

</div>

---

#### Étape 3 — Calculer le vecteur de contexte (somme pondérée)

Le vecteur de contexte $z^{(2)}$ est la **somme pondérée** de tous les vecteurs d'entrée,
chacun multiplié par son poids d'attention :

$$z^{(2)} = \sum_{i} \alpha_{2i} \cdot x^{(i)}$$

```python
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

print(context_vec_2)
```

```
tensor([0.4419, 0.6515, 0.5683])
```

<div align="center">

![Figure 3.10](img/figure_3.10.png)

*Figure 3.10 : $z^{(2)}$ est la somme pondérée de tous les vecteurs d'entrée $x^{(1)}$ à
$x^{(T)}$, pondérés par les poids d'attention correspondants.*

</div>

Contrairement à l'embedding brut de « journey » `[0.55, 0.87, 0.66]`, le vecteur de
contexte $z^{(2)}$ `[0.4419, 0.6515, 0.5683]` encode non seulement le sens de « journey »,
mais aussi **sa relation pondérée avec tous les autres tokens de la phrase**.

La prochaine étape consiste à généraliser ce calcul pour produire simultanément tous les
vecteurs de contexte $z^{(1)}$ à $z^{(T)}$.

---

### 3.3.2 Généralisation — poids d'attention pour tous les tokens

On applique le même pipeline à **tous les tokens simultanément**, en remplaçant les boucles par des opérations matricielles.

<div align="center">

![Figure 3.11](img/figure_3.11.png)

*Figure 3.11 : On étend le calcul de la section précédente à toutes les lignes — un vecteur de contexte $z^{(i)}$ par token.*

</div>

---

#### Étape 1 — Scores d'attention pour toutes les paires

```python
attn_scores = inputs @ inputs.T
print(attn_scores)
```

```
tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
        [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
        ...])
```

> **Pourquoi `inputs @ inputs.T` est équivalent à la double boucle.** `inputs` est une matrice de forme `(6, 3)` — 6 tokens, chacun représenté par un vecteur de dimension 3. `inputs.T` est sa transposée, de forme `(3, 6)`. Leur produit matriciel produit un tenseur `(6, 6)` dont l'élément $(i, j)$ vaut $\sum_k \text{inputs}[i,k] \times \text{inputs}[j,k]$ — exactement le produit scalaire entre les tokens $i$ et $j$, ce que la double boucle calculait explicitement.

#### Étape 2 — Normalisation par softmax

```python
attn_weights = torch.softmax(attn_scores, dim=-1)
```

> **`dim=-1`** indique à PyTorch d'appliquer la softmax le long de la dernière dimension — ici les colonnes. Chaque ligne est normalisée indépendamment, de sorte que ses valeurs somment à 1.

#### Étape 3 — Vecteurs de contexte

```python
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
```

```
tensor([[0.4421, 0.5931, 0.5790],
        [0.4419, 0.6515, 0.5683],
        [0.4431, 0.6496, 0.5671],
        [0.4304, 0.6298, 0.5510],
        [0.4671, 0.5910, 0.5266],
        [0.4177, 0.6503, 0.5645]])
```

`attn_weights` est `(6, 6)` et `inputs` est `(6, 3)` : le produit donne `(6, 3)`, où chaque ligne $i$ est la somme pondérée de tous les vecteurs d'entrée par les poids d'attention du token $i$.

Cette version reste sans paramètres apprenables. La section suivante introduit les matrices $W_Q$, $W_K$, $W_V$ pour rendre ce mécanisme véritablement entraînable.

### 3.4 Self-attention avec poids apprenables

<div align="center">

![Figure 3.13](img/figure_3.13.png)

*Figure 3.13 : On enrichit le mécanisme précédent avec des matrices de poids apprenables.
Les extensions (masque causal, multi-têtes) viendront ensuite.*

</div>

La différence structurelle avec la version simplifiée est l'introduction de **trois matrices de poids apprenables** $W_Q$, $W_K$, $W_V$, mises à jour par rétropropagation. Ce sont elles qui donnent au modèle la capacité d'apprendre *quel type* de similarité est pertinent pour la tâche, plutôt que de mesurer une similarité brute entre embeddings.

---

### 3.4.1 Calcul pas à pas

#### Les trois matrices de projection et leur rôle

Chaque token d'entrée $x^{(i)}$ est projeté dans **trois sous-espaces distincts** via ces matrices :

$$q^{(i)} = x^{(i)} W_Q \qquad k^{(i)} = x^{(i)} W_K \qquad v^{(i)} = x^{(i)} W_V$$

Les termes *query*, *key* et *value* viennent du domaine des bases de données et de la recherche d'information :

- **Query $q^{(i)}$** — la requête émise par le token $i$ pour interroger tous les autres tokens : elle lui permet de trouver, parmi eux, ceux qui sont pertinents pour construire sa représentation contextuelle.

- **Key $k^{(j)}$** — la clé d'indexation du token $j$ : chaque token expose une clé qui sera comparée aux requêtes des autres tokens pour déterminer sa pertinence.

- **Value $v^{(j)}$** — le contenu informationnel réel du token $j$, analogue à la valeur dans une paire clé-valeur d'un dictionnaire : c'est ce qui est effectivement transmis si le token est jugé pertinent.

Le mécanisme procède alors en trois temps : comparer la query $q^{(i)}$ à toutes les keys via le produit scalaire $\omega_{ij} = q^{(i)} \cdot k^{(j)}$, identifier les tokens les plus pertinents via la softmax, puis récupérer les values correspondantes pondérées par ces scores pour former le vecteur de contexte $z^{(i)}$.

C'est un peu comme le fonctionnement d'un moteur de recherche : tu tapes "meilleures pizzerias à Paris" (*query*), Google la compare aux titres et métadonnées de chaque page indexée (*keys*), puis retourne le contenu réel des pages les plus pertinentes (*values*).

Dans la version simplifiée de la section 3.3, $q = k = v = x$ : il n'y avait aucune projection apprise, et la similarité mesurée était la proximité brute entre embeddings. L'introduction de $W_Q$, $W_K$, $W_V$ permet au modèle d'apprendre des représentations spécialisées pour chacun de ces trois rôles.

> **Poids de la matrice vs poids d'attention.** Les éléments de $W_Q$, $W_K$, $W_V$ sont des **paramètres appris** — des scalaires optimisés par descente de gradient, fixes une fois l'entraînement terminé. Les poids d'attention $\alpha_{ij}$ sont eux **dynamiques** : recalculés à chaque forward pass en fonction de l'entrée courante. Ce sont deux usages distincts du mot « poids ».

---

#### Implémentation

```python
x_2 = inputs[1]          # token "journey", shape : (3,)
d_in  = inputs.shape[1]  # 3
d_out = 2                 # dimension de sortie (dans GPT, d_in == d_out)

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
```

`requires_grad=False` désactive ici le calcul du gradient sur ces matrices — uniquement pour alléger les affichages. En entraînement réel, on poserait `requires_grad=True` pour qu'elles soient mises à jour par rétropropagation.

On projette $x^{(2)}$ et l'ensemble des tokens :

```python
query_2 = x_2 @ W_query   # shape : (2,)
keys    = inputs @ W_key   # shape : (6, 2)
values  = inputs @ W_value # shape : (6, 2)

print(query_2)
```

```
tensor([0.4306, 1.4551])
```

On a projeté les 6 tokens de dimension 3 vers un espace de dimension 2. On calcule uniquement `query_2` pour le token courant, mais on a besoin des **keys et values de tous les tokens** pour pondérer leur contribution au vecteur de contexte de $x^{(2)}$.

---

#### Étape 1 — Scores d'attention

<div align="center">

![Figure 3.15](img/figure_3.15.png)

*Figure 3.15 : Les scores sont maintenant calculés entre les projections query/key, et non
plus directement entre les embeddings bruts.*

</div>

```python
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)
```

```
tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])
```

---

#### Étape 2 — Passage aux poids d'attention : le facteur $1/\sqrt{d_k}$

<div align="center">

![Figure 3.16](img/figure_3.16.png)

*Figure 3.16 : Les scores sont mis à l'échelle avant la softmax.*

</div>

```python
d_k = keys.shape[-1]  # dimension des keys = 2
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)
```

```
tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])
```

Tu as raison. Voici la version avec ces termes remplacés par des formulations directes :

---

**Pourquoi diviser par $\sqrt{d_k}$ ?** 

C'est la justification qui donne son nom à l'architecture : *scaled dot-product attention*.

Supposons que les composantes de $q$ et $k$ soient approximativement i.i.d. de loi $\mathcal{N}(0, 1)$. Le produit scalaire $q \cdot k = \sum_{l=1}^{d_k} q_l k_l$ est alors une somme de $d_k$ variables aléatoires indépendantes de moyenne 0 et de variance 1, donc :

$$\text{Var}(q \cdot k) = d_k \qquad \Rightarrow \qquad \sigma(q \cdot k) = \sqrt{d_k}$$

<details> <summary>Démonstration</summary> 

**Posons** $q_l \sim \mathcal{N}(0,1)$ et $k_l \sim \mathcal{N}(0,1)$, indépendantes.

**Étape 1 — Variance du produit $q_l k_l$**

$$\text{Var}(q_l k_l) = \mathbb{E}[q_l^2 k_l^2] - \mathbb{E}[q_l k_l]^2$$

Puisque $q_l, k_l \sim \mathcal{N}(0,1)$ indépendantes, leurs carrés suivent des lois du khi-deux à 1 degré de liberté indépendantes :

$$q_l^2 \sim \chi^2(1) \qquad k_l^2 \sim \chi^2(1) \quad \text{indépendantes}$$

Pour une loi $\chi^2(1)$ : $\mathbb{E}[X] = 1$. Par indépendance de $q_l^2$ et $k_l^2$ :

$$\mathbb{E}[q_l^2 k_l^2] = \mathbb{E}[q_l^2]\,\mathbb{E}[k_l^2] = 1 \times 1 = 1$$

Et $\mathbb{E}[q_l k_l] = \mathbb{E}[q_l]\,\mathbb{E}[k_l] = 0$, donc :

$$\text{Var}(q_l k_l) = 1 - 0 = 1$$

**Étape 2 — Variance de la somme $q \cdot k = \sum_{l=1}^{d_k} q_l k_l$**

Les termes $q_l k_l$ sont indépendants entre eux (les indices $l$ sont distincts), donc la variance de leur somme est la somme de leurs variances :

$$\text{Var}(q \cdot k) = \sum_{l=1}^{d_k} \text{Var}(q_l k_l) = \sum_{l=1}^{d_k} 1 = d_k$$

**Étape 3 — Écart-type**

$$\sigma(q \cdot k) = \sqrt{\text{Var}(q \cdot k)} = \sqrt{d_k}$$

**Étape 4 — Après mise à l'échelle par $\sqrt{d_k}$**

$$\text{Var}\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{1}{(\sqrt{d_k})^2}\,\text{Var}(q \cdot k) = \frac{d_k}{d_k} = 1$$

La variance est ramenée à 1 quelle que soit la valeur de $d_k$. $\blacksquare$    

</details>

<br>

Pour $d_k = 1024$ (typique dans GPT), les produits scalaires ont un écart-type de l'ordre de 32. La softmax reçoit donc des entrées très étalées : les grandes valeurs poussent vers 1, les petites vers 0, et la distribution ressemble à un vecteur one-hot.

Or le gradient de la softmax est $w_i(1 - w_i)$ : quand $w_i \to 1$ ou $w_i \to 0$, ce gradient tend vers 0. On retombe dans le problème du gradient évanescent — l'entraînement stagne.

En divisant par $\sqrt{d_k}$, on ramène la variance du score à 1 :

$$\text{Var}\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

Les entrées de la softmax restent dans des plages raisonnables quel que soit $d_k$ — les poids d'attention ne s'écrasent plus vers 0 ou 1, et les gradients restent exploitables.

---

#### Étape 3 — Vecteur de contexte

<div align="center">

![Figure 3.17](img/figure_3.17.png)

*Figure 3.17 : Le vecteur de contexte est la somme pondérée des vecteurs value — et non
plus des embeddings bruts.*

</div>

```python
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)
```

```
tensor([0.3061, 0.8210])
```

La différence fondamentale avec la section 3.3 : on somme les **vecteurs value** $v^{(j)}$, pas les embeddings bruts $x^{(j)}$. $W_V$ permet au modèle d'apprendre quelle information extraire de chaque token pour la transmettre — indépendamment de la façon dont ce token est indexé (key) ou de la façon dont il interroge les autres (query).

Explicitement, le vecteur de contexte de $x^{(2)}$ s'écrit :

$$z^{(2)} = \sum_{j=1}^{T} \alpha_{2j} \cdot v^{(j)} = \sum_{j=1}^{T} \alpha_{2j} \cdot x^{(j)} W_V$$

En forme matricielle compacte, pour l'ensemble des tokens simultanément :

$$Z = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

où $Q = X W_Q$, $K = X W_K$, $V = X W_V$ sont les projections de toute la séquence, et chaque ligne $i$ de $Z$ est le vecteur de contexte $z^{(i)}$.

---
## 3.4.2 Encapsulation en classe Python

Le code précédent est réorganisé en une classe `nn.Module` — la brique de base de tout modèle PyTorch, qui gère automatiquement l'enregistrement des paramètres, leur mise à jour lors de l'entraînement, et le déplacement sur GPU.

```python
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys    = x @ self.W_key
        queries = x @ self.W_query
        values  = x @ self.W_value
        attn_scores  = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec  = attn_weights @ values
        return context_vec
```

```python
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))
```

```
tensor([[0.2996, 0.8053],
        [0.3061, 0.8210],
        [0.3058, 0.8203],
        [0.2948, 0.7939],
        [0.2927, 0.7891],
        [0.2990, 0.8040]], grad_fn=<MmBackward0>)
```

<div align="center">

![Figure 3.18](img/figure_3.18.png)

*Figure 3.18 : Résumé matriciel de la self-attention. $X$ est projeté en $Q$, $K$, $V$ via les trois matrices de poids. Les scores $QK^\top / \sqrt{d_k}$ sont normalisés par softmax, puis multipliés par $V$ pour produire $Z$.*

</div>

---

### Version améliorée : `SelfAttention_v2` avec `nn.Linear`

**Initialisation des poids.** Dans `SelfAttention_v1`, `nn.Parameter(torch.rand(...))` tire chaque poids indépendamment selon $W_{ij} \sim \mathcal{U}[0, 1)$. Cette initialisation est naïve pour deux raisons :

- Les poids initiaux sont tous **positifs**, ce qui restreint la dynamique d'apprentissage. Une bonne initialisation doit au contraire différencier efficacement les neurones dès le départ en assignant des poids aléatoires de signes variés (on parle de **Symmetry breaking** dans la littérature);
  
<br>
  
- la variance $\operatorname{Var}(W_{ij}) = \frac{1}{12}$ est indépendante de $d_{in}$, ce qui fait exploser la variance de la sortie $XW$ quand $d_{in}$ est grand.

<details>
<summary>Démonstration</summary>

$K = XW$ est la projection des embeddings d'entrée dans l'espace key, avec $K \in \mathbb{R}^{n \times d_{out}}$ : chaque ligne est le vecteur key d'un token, chaque colonne est une dimension de l'espace key. On s'intéresse au scalaire :

$$y = K_{i,j} = \sum_{k=1}^{d_{in}} x_{i,k}\, W_{k,j}$$

la $j$-ième composante du vecteur key du token $i$. C'est cette somme dont on analyse la variance pour quantifier l'effet de l'initialisation.

**Hypothèses de travail.** On suppose $x_k$ et $W_k$ indépendants, $\mathbb{E}[x_k] = 0$ et $\operatorname{Var}(x_k) = \sigma^2$ — hypothèse standard vérifiée après normalisation des entrées.

**Variance d'un terme $x_k W_k$.** Avec $\mathbb{E}[x_k] = 0$ :

$$\operatorname{Var}(x_k W_k) = \mathbb{E}[x_k^2 W_k^2] - \mathbb{E}[x_k W_k]^2 = \mathbb{E}[x_k^2]\,\mathbb{E}[W_k^2] - \underbrace{\mathbb{E}[x_k]^2}_{=\,0}\mathbb{E}[W_k]^2 = \sigma^2\,\mathbb{E}[W_k^2]$$

La quantité clé est $\mathbb{E}[W_k^2]$, qui se décompose comme :

$$\mathbb{E}[W_k^2] = \operatorname{Var}(W_k) + \mathbb{E}[W_k]^2$$

**Rappel : loi uniforme $\mathcal{U}[a,b]$.**

$$\operatorname{Var}(W) = \frac{(b-a)^2}{12}, \qquad \mathbb{E}[W] = \frac{a+b}{2}$$

**Cas `torch.rand` : $W_k \sim \mathcal{U}[0, 1)$.**

$$\mathbb{E}[W_k] = \frac{1}{2}, \qquad \operatorname{Var}(W_k) = \frac{1}{12}$$

$$\mathbb{E}[W_k^2] = \frac{1}{12} + \frac{1}{4} = \frac{1}{3}$$

$$\operatorname{Var}(y) = \sum_{k=1}^{d_{in}} \sigma^2 \cdot \frac{1}{3} = \frac{d_{in}\,\sigma^2}{3}$$

La variance croît **linéairement** avec $d_{in}$ — pour $d_{in} = 512$, elle est 512 fois plus grande que la variance d'entrée.

**Cas Kaiming : $W_k \sim \mathcal{U}\!\left(-\dfrac{1}{\sqrt{d_{in}}}, \dfrac{1}{\sqrt{d_{in}}}\right)$.**

$$\mathbb{E}[W_k] = 0, \qquad \operatorname{Var}(W_k) = \frac{\left(\frac{2}{\sqrt{d_{in}}}\right)^2}{12} = \frac{1}{3\,d_{in}}$$

$$\mathbb{E}[W_k^2] = \frac{1}{3\,d_{in}} + 0 = \frac{1}{3\,d_{in}}$$

$$\operatorname{Var}(y) = \sum_{k=1}^{d_{in}} \sigma^2 \cdot \frac{1}{3\,d_{in}} = \frac{\sigma^2}{3}$$

Le $d_{in}$ se simplifie — la variance reste constante quelle que soit la dimension d'entrée.

</details>

`nn.Linear` utilise par défaut l'**initialisation de Kaiming** (He, 2015), qui fixe :

$$W_{ij} \sim \mathcal{U}\!\left(-\frac{1}{\sqrt{d_{in}}},\ \frac{1}{\sqrt{d_{in}}}\right) \qquad \Longrightarrow \qquad \operatorname{Var}(W_{ij}) = \frac{1}{3\,d_{in}}$$

ce qui maintient la variance du signal constante à travers les couches, quelle que soit la dimension.

---

**`nn.Linear` comme opérateur linéaire.** Dans `SelfAttention_v1`, on définit $W \in \mathbb{R}^{d_{in} \times d_{out}}$ comme un `nn.Parameter`, et le forward calcule :

$$K = XW, \qquad X \in \mathbb{R}^{n \times d_{in}},\quad K \in \mathbb{R}^{n \times d_{out}}$$

`nn.Linear(d_in, d_out, bias=False)` stocke en interne une matrice $\widetilde{W} \in \mathbb{R}^{d_{out} \times d_{in}}$, et son appel sur $X$ calcule :

$$K = X\,\widetilde{W}^\top$$

Les deux expressions sont identiques si et seulement si $\widetilde{W} = W^\top$ — ce qui est exactement la convention de stockage de `nn.Linear`. Les deux implémentations sont donc **strictement équivalentes**.

```python
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)
        attn_scores  = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec  = attn_weights @ values
        return context_vec
```

```python
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))
```

```
tensor([[-0.0739,  0.0713],
        [-0.0748,  0.0703],
        [-0.0749,  0.0702],
        [-0.0760,  0.0685],
        [-0.0763,  0.0679],
        [-0.0754,  0.0693]], grad_fn=<MmBackward0>)
```

Les sorties diffèrent entre `v1` et `v2` uniquement parce que les poids initiaux sont différents — la logique du forward pass est identique.

---

> **Exercice 3.1.** Transférer les poids de `SelfAttention_v2` vers `SelfAttention_v1` et vérifier que les deux implémentations produisent les mêmes sorties sur `inputs`.

---

La prochaine étape enrichit ce mécanisme de deux extensions : le **masque causal**, qui empêche chaque token d'accéder aux tokens futurs lors de la génération, et l'**attention multi-têtes**, qui fait tourner plusieurs mécanismes d'attention en parallèle pour capter différents types de relations entre tokens.

---
## 3.5 Masquer les tokens futurs avec l'attention causale

L'attention causale (ou *masked attention*) est une variante de l'auto-attention qui contraint chaque token à n'accéder qu'aux tokens qui le précèdent (et à lui-même) dans la séquence. C'est l'opposé de l'auto-attention standard qui dispose de toute la séquence en entrée.

Cette contrainte est fondamentale pour les LLMs de type GPT : lors de la prédiction du token suivant, le modèle ne doit pas « voir » les tokens futurs — ce serait une fuite d'information.

---

## 3.5.1 Appliquer un masque d'attention causale

Il existe deux approches équivalentes pour obtenir la matrice de poids d'attention masquée.

### Approche naïve : masquer après softmax

**Étape 1 — Calculer les poids d'attention standards (softmax)**

```python
queries = sa_v2.W_query(inputs)
keys    = sa_v2.W_key(inputs)
attn_scores   = queries @ keys.T
attn_weights  = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
```
```
tensor([[0.1921, 0.1646, 0.1652, 0.1550, 0.1721, 0.1510],
        [0.2041, 0.1659, 0.1662, 0.1496, 0.1665, 0.1477],
        [0.2036, 0.1659, 0.1662, 0.1498, 0.1664, 0.1480],
        [0.1869, 0.1667, 0.1668, 0.1571, 0.1661, 0.1564],
        [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.1585],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<SoftmaxBackward0>)
```

**Étape 2 — Construire le masque triangulaire inférieur**

```python
context_length = attn_scores.shape[0]
mask_simple    = torch.tril(torch.ones(context_length, context_length))
```
```
tensor([[1., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1.]])
```

`torch.tril` conserve la diagonale et le triangle inférieur, et met à zéro le triangle supérieur — exactement les positions futures à masquer.

**Étape 3 — Zeroing : multiplier les poids par le masque**

```python
masked_simple = attn_weights * mask_simple
```
```
tensor([[0.1921, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2041, 0.1659, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2036, 0.1659, 0.1662, 0.0000, 0.0000, 0.0000],
        [0.1869, 0.1667, 0.1668, 0.1571, 0.0000, 0.0000],
        [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<MulBackward0>
```

Les positions futures valent maintenant 0, mais les lignes ne somment plus à 1 — la distribution de probabilité est rompue.

**Étape 4 — Renormaliser**

```python
row_sums          = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
```
```
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
        [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
        [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<DivBackward0>)
```

Chaque ligne somme à 1. Les poids masqués sont nuls, et les poids restants sont redistribués proportionnellement.

---

> **Note — Pas de fuite d'information**
> On pourrait craindre que les tokens futurs aient déjà « contaminé » le résultat via le softmax initial, avant d'être mis à zéro. Ce n'est pas le cas. La renormalisation après masquage est mathématiquement équivalente à avoir calculé le softmax uniquement sur les positions non-masquées dès le départ. Les tokens futurs n'exercent aucune influence sur la distribution finale.

<details>
<summary>Démonstration</summary>

### Cadre et notations

Soit une séquence de $T$ tokens. À chaque token en position $j$ sont associés trois vecteurs obtenus par projections linéaires de son embedding :

- $q_j \in \mathbb{R}^{d_k}$ : vecteur **requête** (*query*)
- $k_j \in \mathbb{R}^{d_k}$ : vecteur **clé** (*key*)
- $v_j \in \mathbb{R}^{d_v}$ : vecteur **valeur** (*value*)

Pour un token courant en position $i$, le **score d'attention brut** avec chaque token en position $j$ est :

$$e_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}} \in \mathbb{R}, \qquad j \in \{1, \dots, T\}$$

La **sortie** du mécanisme d'attention pour le token $i$ est :

$$z_i = \sum_{j=1}^{T} \alpha_{ij}\, v_j, \qquad \alpha_{ij} \geq 0, \quad \sum_{j=1}^{T} \alpha_{ij} = 1$$

---

### Méthode 1 — Softmax global, puis masquage et renormalisation

**Étape 1 — Softmax sur l'ensemble des tokens.**

On calcule les poids d'attention sans aucune restriction :

$$\alpha_{ij} = \frac{e^{e_{ij}}}{\displaystyle\sum_{m=1}^{T} e^{e_{im}}}$$

Tous les tokens contribuent au dénominateur, y compris les tokens futurs.

**Étape 2 — Application du masque triangulaire.**

On annule les poids correspondant aux tokens futurs :

$$\tilde{\alpha}_{ij} = \begin{cases} \alpha_{ij} & \text{si } j \leq i \\ 0 & \text{si } j > i \end{cases}$$

Après cette étape, les poids ne somment plus à 1 : on a retiré de la masse de probabilité.

**Étape 3 — Renormalisation.**

On divise chaque poids par la somme de la ligne :

$$\alpha_{ij}^{(1)} = \frac{\tilde{\alpha}_{ij}}{\displaystyle\sum_{k=1}^{T} \tilde{\alpha}_{ik}}$$

**Développement explicite.** On simplifie séparément le numérateur et le dénominateur.

Au numérateur, on distingue deux cas selon la position de $j$ par rapport à $i$, en appliquant successivement la définition de $\tilde{\alpha}_{ij}$ (Étape 2) puis celle de $\alpha_{ij}$ (Étape 1) :

$$\tilde{\alpha}_{ij} = \begin{cases} \alpha_{ij} = \dfrac{e^{e_{ij}}}{\displaystyle\sum_{m=1}^{T} e^{e_{im}}} & \text{si } j \leq i \\[10pt] 0 & \text{si } j > i \end{cases}$$

Au dénominateur, on développe $\displaystyle\sum_{k=1}^{T} \tilde{\alpha}_{ik}$ en séparant les positions visibles ($k \leq i$) et futures ($k > i$), puis on applique successivement la définition de $\tilde{\alpha}_{ik}$ (Étape 2) puis celle de $\alpha_{ik}$ (Étape 1) :

$$\sum_{k=1}^{T} \tilde{\alpha}_{ik} = \sum_{k \leq i} \underbrace{\tilde{\alpha}_{ik}}_{=\,\alpha_{ik}} + \sum_{k > i} \underbrace{\tilde{\alpha}_{ik}}_{=\,0} = \sum_{k \leq i} \alpha_{ik} = \sum_{k \leq i} \frac{e^{e_{ik}}}{\displaystyle\sum_{m=1}^{T} e^{e_{im}}}$$

Le terme $\displaystyle\sum_{m=1}^{T} e^{e_{im}}$ est une constante par rapport à $k$, on peut le factoriser :

$$\sum_{k=1}^{T} \tilde{\alpha}_{ik} = \frac{\displaystyle\sum_{k \leq i} e^{e_{ik}}}{\displaystyle\sum_{m=1}^{T} e^{e_{im}}}$$

On substitue numérateur et dénominateur dans l'expression de $\alpha_{ij}^{(1)}$ pour $j \leq i$ :

$$\alpha_{ij}^{(1)} = \frac{\tilde{\alpha}_{ij}}{\displaystyle\sum_{k=1}^{T} \tilde{\alpha}_{ik}} = \frac{\quad\dfrac{e^{e_{ij}}}{\displaystyle\sum_{m=1}^{T} e^{e_{im}}}\quad}{\dfrac{\displaystyle\sum_{k \leq i} e^{e_{ik}}}{\displaystyle\sum_{m=1}^{T} e^{e_{im}}}}$$

Le facteur $\displaystyle\sum_{m=1}^{T} e^{e_{im}}$ est présent au numérateur **et** au dénominateur : il se simplifie exactement :

$$\alpha_{ij}^{(1)} = \frac{e^{e_{ij}}}{\displaystyle\sum_{m=1}^{T} e^{e_{im}}} \times \frac{\displaystyle\sum_{m=1}^{T} e^{e_{im}}}{\displaystyle\sum_{k \leq i} e^{e_{ik}}}$$

$$\boxed{\alpha_{ij}^{(1)} = \frac{e^{e_{ij}}}{\displaystyle\sum_{k \leq i} e^{e_{ik}}} \quad \text{pour } j \leq i, \qquad \alpha_{ij}^{(1)} = 0 \quad \text{pour } j > i}$$

---

### Méthode 2 — Masquage des scores bruts avant le softmax

**Étape 1 — Masquage des scores bruts.**

On remplace les scores des tokens futurs par $-\infty$ avant tout calcul :

$$\tilde{e}_{ij} = \begin{cases} e_{ij} & \text{si } j \leq i \\ -\infty & \text{si } j > i \end{cases}$$

**Étape 2 — Softmax unique sur les scores masqués.**

$$\alpha_{ij}^{(2)} = \frac{e^{\,\tilde{e}_{ij}}}{\displaystyle\sum_{k=1}^{T} e^{\,\tilde{e}_{ik}}}$$

**Développement explicite.** On décompose le dénominateur en séparant les positions visibles et futures, puis on applique la définition de $\tilde{e}_{ik}$ :

$$\sum_{k=1}^{T} e^{\,\tilde{e}_{ik}} = \sum_{k \leq i} \underbrace{e^{\,\tilde{e}_{ik}}}_{=\,e^{e_{ik}}} + \sum_{k > i} \underbrace{e^{\,\tilde{e}_{ik}}}_{=\,e^{-\infty}\,=\,0} = \sum_{k \leq i} e^{e_{ik}}$$

De même au numérateur, en appliquant la définition de $\tilde{e}_{ij}$ :

$$e^{\,\tilde{e}_{ij}} = \begin{cases} e^{e_{ij}} & \text{si } j \leq i \\ e^{-\infty} = 0 & \text{si } j > i \end{cases}$$

On obtient donc :

$$\boxed{\alpha_{ij}^{(2)} = \frac{e^{e_{ij}}}{\displaystyle\sum_{k \leq i} e^{e_{ik}}} \quad \text{pour } j \leq i, \qquad \alpha_{ij}^{(2)} = 0 \quad \text{pour } j > i}$$

---

### Équivalence des deux méthodes

La comparaison terme à terme des deux résultats encadrés est immédiate :

$$\text{Pour } j \leq i : \quad \alpha_{ij}^{(1)} = \frac{e^{e_{ij}}}{\displaystyle\sum_{k \leq i} e^{e_{ik}}} = \alpha_{ij}^{(2)}$$

$$\text{Pour } j > i : \quad \alpha_{ij}^{(1)} = 0 = \alpha_{ij}^{(2)}$$

On conclut :

$$\boxed{\alpha_{ij}^{(1)} = \alpha_{ij}^{(2)} \qquad \forall\, j \in \{1, \dots, T\}}$$

Il s'agit d'une **égalité algébrique exacte**. Dans la Méthode 1, le facteur $\displaystyle\sum_{m=1}^{T} e^{e_{im}}$ introduit par le premier softmax s'annule exactement lors de la renormalisation. La présence initiale des tokens futurs dans le calcul n'a laissé **aucune trace** dans le résultat final : le dénominateur ne contient plus que $\displaystyle\sum_{k \leq i} e^{e_{ik}}$, exactement comme dans la Méthode 2 où les tokens futurs n'ont jamais été inclus. Il n'y a donc **aucune fuite d'information**.

</details>

---

### Approche efficace : masquer avant softmax avec $-\infty$

L'approche naïve applique softmax puis corrige. On peut faire mieux : masquer les scores d'attention **avant** le softmax en remplaçant les positions futures par $-\infty$.

La justification est immédiate : $e^{-\infty} = 0$, donc softmax attribue automatiquement un poids nul à ces positions, et la somme des poids restants vaut 1 — sans aucune renormalisation manuelle.

```python
mask   = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
```

`torch.triu` avec `diagonal=1` isole strictement le triangle **supérieur** (hors diagonale) — les positions futures. `masked_fill` remplace ces positions par $-\infty$.

```
tensor([[0.2899,   -inf,   -inf,   -inf,   -inf,   -inf],
        [0.4656, 0.1723,   -inf,   -inf,   -inf,   -inf],
        [0.4594, 0.1703, 0.1731,   -inf,   -inf,   -inf],
        [0.2642, 0.1024, 0.1036, 0.0186,   -inf,   -inf],
        [0.2183, 0.0874, 0.0882, 0.0177, 0.0786,   -inf],
        [0.3408, 0.1270, 0.1290, 0.0198, 0.1290, 0.0078]],
       grad_fn=<MaskedFillBackward0>)
```

Il suffit ensuite d'appliquer softmax :

```python
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
```
```
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
        [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
        [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<SoftmaxBackward0>)
```

Le résultat est identique à l'approche naïve, en deux étapes au lieu de quatre. C'est cette approche qui est retenue en pratique.

---
## 3.5.2 Masquer des poids d'attention supplémentaires avec le dropout

Le dropout est une technique de régularisation proposée par Geoffrey Hinton en 2012. À chaque étape d'entraînement, chaque neurone a une probabilité $p$ (le *taux d'abandon*) d'être temporairement désactivé — ignoré pour cette passe, mais potentiellement actif à la suivante. Après l'entraînement, aucun neurone n'est jamais désactivé.

L'intuition derrière cette technique peut sembler paradoxale. Imaginez une entreprise où chaque employé joue à pile ou face chaque matin pour décider s'il vient travailler. L'entreprise serait forcée de s'adapter : elle ne pourrait plus compter sur une seule personne pour une tâche critique, toutes les expertises devraient être distribuées, les employés apprendraient à collaborer avec de nombreux collègues plutôt qu'une poignée fixe. Elle deviendrait bien plus robuste. C'est exactement ce qui se passe dans un réseau de neurones : les neurones entraînés sous contrainte de dropout ne peuvent pas s'adapter de concert avec leurs voisins habituels — ils développent chacun une plus grande utilité propre et deviennent moins sensibles aux légères variations en entrée.

Une autre façon de voir les choses : à chaque étape d'entraînement, le réseau actif est différent (parmi $2^N$ configurations possibles pour $N$ neurones). Le réseau final peut être vu comme un ensemble moyen de tous ces sous-réseaux.

Dans le mécanisme d'attention, le dropout s'applique directement sur les poids d'attention — la variante la plus courante en pratique. Concrètement, le masque causal et le masque de dropout se superposent : le premier annule le triangle supérieur (tokens futurs), le second annule aléatoirement des positions supplémentaires parmi les tokens visibles.

<div align="center">

![Figure 3.22](img/figure_3.22.png)

*Figure 3.22 : À partir de la matrice d'attention causale (triangle supérieur nul), un masque de dropout aléatoire est appliqué pour annuler des poids supplémentaires parmi les positions visibles, réduisant le risque de sur-apprentissage pendant l'entraînement.*

</div>

---

### Mise à l'échelle automatique

Avec un taux de dropout $p$, les poids survivants sont multipliés par $\frac{1}{1-p}$. C'est une correction nécessaire : pendant l'entraînement, un neurone n'est connecté en moyenne qu'à une fraction $(1-p)$ de ses entrées habituelles. Sans cette compensation, le neurone recevrait à l'inférence (où tous les neurones sont actifs) un signal d'une amplitude très différente de ce qu'il a appris.

On vérifie ce comportement sur une matrice de 1s avec $p = 0.5$ :

```python
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print(dropout(example))
```
```
tensor([[2., 2., 0., 2., 2., 0.],
        [0., 0., 0., 2., 0., 2.],
        [2., 2., 2., 2., 0., 2.],
        [0., 2., 2., 0., 0., 2.],
        [0., 2., 0., 2., 0., 2.],
        [0., 2., 2., 2., 2., 0.]])
```


Les éléments restés non nuls après le dropout valent $\frac{1}{1 - 0.5} = 2$, confirmant la mise à l'échelle.

---

### Application aux poids d'attention causale

```python
torch.manual_seed(123)
print(dropout(attn_weights))
```
```
tensor([[2.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.7599, 0.6194, 0.6206, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.4921, 0.4925, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.3966, 0.0000, 0.3775, 0.0000, 0.0000],
        [0.0000, 0.3327, 0.3331, 0.3084, 0.3331, 0.0000]],
       grad_fn=<MulBackward0>)
```

Le masque causal (triangle supérieur nul) est préservé. Le dropout vient s'y superposer en annulant aléatoirement des poids supplémentaires parmi les positions visibles, et en rescalant les survivants.

> **Note** — Les résultats du dropout peuvent varier selon le système d'exploitation, indépendamment du `manual_seed`. Ce comportement est documenté dans le [suivi de problèmes PyTorch](https://github.com/pytorch/pytorch/issues/121595).
>
---
Tu as raison sur les deux points. Le troisième "problème" que j'avais listé n'en était pas un distinct — c'était une reformulation du premier. Voici la section corrigée et les explications approfondies.

---

## 3.5.3 Implémenter une classe d'attention causale compacte

On intègre maintenant le masque causal et le dropout dans une classe `CausalAttention` qui remplace `SelfAttention`. Cette classe doit également gérer des entrées en batch — plusieurs séquences traitées en parallèle.

Pour simuler un batch, on duplique l'entrée :

```python
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)
# torch.Size([2, 6, 3])
```

Un tenseur 3D de forme `(batch_size, num_tokens, d_in)` : 2 séquences, 6 tokens chacune, embeddings de dimension 3.

---

### Le code

```python
class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec
```

```python
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)
# context_vecs.shape: torch.Size([2, 6, 2])
```

---

### Point 1 — `register_buffer` : qu'est-ce que c'est et pourquoi ?

Dans un `nn.Module` PyTorch, il existe deux types de tenseurs :

**Les paramètres** (`nn.Parameter`) — les poids appris, enregistrés via `nn.Linear`. PyTorch les suit automatiquement : ils apparaissent dans `model.parameters()`, reçoivent des gradients, et sont déplacés sur GPU avec `model.to(device)`.

**Les buffers** (`register_buffer`) — des tenseurs fixes, non appris, dont le modèle a besoin mais qui ne doivent pas être mis à jour par l'optimiseur. C'est exactement le cas du masque causal : il est constant, calculé une seule fois à l'initialisation.

Sans `register_buffer`, on pourrait écrire directement `self.mask = torch.triu(...)`. Ça fonctionnerait en local — mais avec deux problèmes :

Le masque ne serait **pas déplacé automatiquement sur GPU**. Si on fait `model.to("cuda")`, les poids migrent, mais pas `self.mask`. Au moment du `masked_fill_`, PyTorch lèverait une erreur de device mismatch : le masque est sur CPU, les scores d'attention sur GPU.

Le masque **n'apparaîtrait pas dans `model.state_dict()`**, le dictionnaire qui capture l'état complet du modèle pour la sauvegarde. Un modèle rechargé depuis un checkpoint n'aurait plus son masque.

`register_buffer` règle les deux : le masque suit le modèle partout (CPU/GPU) et est inclus dans `state_dict()`.

---

### Point 2 — Anatomie du code de masquage

#### `register_buffer('mask', torch.triu(...))`

L'appel `register_buffer('mask', tenseur)` fait deux choses : il enregistre le tenseur comme buffer du module, et il le rend accessible via `self.mask`. Le premier argument `'mask'` est simplement le nom sous lequel le buffer est enregistré — c'est ce nom qui devient l'attribut. On aurait pu écrire `register_buffer('causal_mask', ...)` et y accéder ensuite via `self.causal_mask`.

Le tenseur enregistré est :

```python
torch.triu(torch.ones(context_length, context_length), diagonal=1)
```

`torch.triu` extrait le triangle supérieur d'une matrice. Avec `diagonal=1`, la diagonale principale est exclue — seules les positions strictement au-dessus valent 1 :

```
Mask with diagonal=1:
 tensor([[0., 1., 1., 1., 1., 1.],
        [0., 0., 1., 1., 1., 1.],
        [0., 0., 0., 1., 1., 1.],
        [0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0.]])
```

Les 1 marquent exactement les positions futures à masquer.

#### `attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)`

Décomposons paramètre par paramètre.

`self.mask.bool()` — convertit le masque de `float` (0.0 / 1.0) en `bool` (False / True). `masked_fill_` attend un masque booléen : les positions à `True` seront remplacées.

`[:num_tokens, :num_tokens]` — le masque a été créé à l'initialisation avec la taille maximale `context_length × context_length`. La séquence courante peut être plus courte. Ce slicing extrait le sous-masque de la taille exacte de la séquence traitée, sans recréer le tenseur à chaque appel.

`-torch.inf` — la valeur de remplacement pour les positions futures. Comme $e^{-\infty} = 0$, le softmax leur attribuera un poids nul.

`masked_fill_` (avec underscore) — opération in-place : `attn_scores` est modifié directement en mémoire sans copie intermédiaire, ce qui économise de la mémoire pour des matrices d'attention potentiellement grandes.

En résumé : pour chaque position $(i, j)$ où `self.mask[i,j]` vaut `True` (c'est-à-dire $j > i$, un token futur), la valeur correspondante dans `attn_scores` est remplacée par $-\infty$.

---

### Point 3 — `keys.transpose(1, 2)` au lieu de `keys.T`

Dans la section 3.5.1, on avait `keys.T` car les entrées étaient 2D `(num_tokens, d_out)`. Ici `keys` est 3D : `(b, num_tokens, d_out)`. `.T` inverserait tous les axes — on obtiendrait `(d_out, num_tokens, b)`, ce qui est faux. `.transpose(1, 2)` n'échange que les deux dernières dimensions, donnant `(b, d_out, num_tokens)` : la dimension batch reste en position 0, et le produit matriciel opère correctement sur chaque séquence du batch indépendamment.

---

<div align="center">

![Figure 3.23](img/figure_3.23.png)

*Figure 3.23 : Progression de l'implémentation — de l'attention simplifiée à l'attention causale avec poids entraînables, masque causal et dropout. La prochaine étape est l'attention multi-têtes.*

</div>

---
## 3.6 Étendre l'attention mono-tête à l'attention multi-têtes

L'attention multi-têtes consiste à faire tourner le mécanisme d'attention plusieurs fois en parallèle, chacune avec ses propres matrices de poids $W_Q$, $W_K$, $W_V$. Chaque tête apprend à se spécialiser sur un type de relation différent dans la séquence. Un module `CausalAttention` seul correspond à l'attention mono-tête — un seul ensemble de poids traitant l'entrée.

---

## 3.6.1 Empiler plusieurs modules d'attention mono-tête

### L'idée

La façon la plus directe d'implémenter l'attention multi-têtes est d'instancier plusieurs `CausalAttention` indépendants, de les faire tourner sur la même entrée, et de concaténer leurs sorties.

<div align="center">

![Figure 3.24](img/figure_3.24.png)

*Figure 3.24 : Un module d'attention multi-têtes avec deux têtes. Chaque tête dispose de ses propres matrices de poids $W_Q$, $W_K$, $W_V$, et produit ses propres vecteurs de contexte $Z_1$ et $Z_2$, ensuite concaténés en $Z$.*

</div>

### Implémentation

```python
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
```

Deux points méritent attention.

`nn.ModuleList` est une liste de modules `CausalAttention` dont les paramètres sont centralisés au niveau de `MultiHeadAttentionWrapper` — accessibles via `model.parameters()`, déplacés avec `model.to(device)`, et sauvegardés dans `state_dict()`.

`torch.cat(..., dim=-1)` concatène les sorties de chaque tête le long de la dernière dimension — la dimension d'embedding. Si chaque tête produit des vecteurs de dimension `d_out`, la sortie finale a dimension `d_out × num_heads`.

### Exemple

```python
torch.manual_seed(123)
context_length = batch.shape[1]
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```
```
tensor([[[-0.4519,  0.2216,  0.4772,  0.1063],
         [-0.5874,  0.0058,  0.5891,  0.3257],
         [-0.6300, -0.0632,  0.6202,  0.3860],
         [-0.5675, -0.0843,  0.5478,  0.3589],
         [-0.5526, -0.0981,  0.5321,  0.3428],
         [-0.5299, -0.1081,  0.5077,  0.3493]],

        [[-0.4519,  0.2216,  0.4772,  0.1063],
         [-0.5874,  0.0058,  0.5891,  0.3257],
         [-0.6300, -0.0632,  0.6202,  0.3860],
         [-0.5675, -0.0843,  0.5478,  0.3589],
         [-0.5526, -0.0981,  0.5321,  0.3428],
         [-0.5299, -0.1081,  0.5077,  0.3493]]], grad_fn=<CatBackward0>)
context_vecs.shape torch.Size([2, 6, 4])
```

La forme `(2, 6, 4)` se lit : 2 séquences dans le batch, 6 tokens chacune, vecteurs de contexte de dimension $2 \times 2 = 4$. Les deux séquences sont identiques (batch simulé par duplication), d'où des vecteurs de contexte identiques.

<div align="center">

![Figure 3.25](img/figure_3.25.png)

*Figure 3.25 : Avec `num_heads=2` et `d_out=2`, chaque tête produit une matrice de vecteurs de contexte de dimension 2. Les deux matrices sont concaténées le long de la dimension des colonnes, donnant une dimension finale de $2 \times 2 = 4$.*

</div>

> **Exercice 3.2** — Pour obtenir des vecteurs de contexte de dimension 2 avec `num_heads=2`, il suffit de passer `d_out=1` : chaque tête produit des vecteurs de dimension 1, et leur concaténation donne bien la dimension 2.

---

### Limite de cette approche

Les têtes sont traitées **séquentiellement** dans `forward` via `[head(x) for head in self.heads]`. C'est fonctionnellement correct mais inefficace : on effectue `num_heads` passes distinctes là où une seule opération matricielle pourrait tout calculer en parallèle. La section suivante présente une implémentation qui exploite cette parallélisation.

---
## 3.6.2 Implémentation efficace par découpe de poids

La classe `MultiHeadAttention` fusionne `MultiHeadAttentionWrapper` et
`CausalAttention` en une seule entité. L'idée directrice : effectuer **une seule**
projection linéaire de dimension $d_{out}$, puis découper implicitement ce résultat
en $H = \texttt{num\_heads}$ sous-espaces de dimension $d_h = d_{out}/H$.
Contrairement à `MultiHeadAttentionWrapper`, qui maintenait $H$ matrices de poids
$W_{K,j} \in \mathbb{R}^{d_h \times d_{in}}$ et répétait la multiplication matricielle
pour chaque tête, une seule multiplication suffit ici.

<div align="center">

![mha_explained_1](img/mha_explained/mha_explained_1.png)
*Figure X : Vue d'ensemble du mécanisme d'attention multi-têtes (source : CNRS-FIDLE).*

</div>

Nous décrivons ci-dessous les transformations successives appliquées au tenseur des
clés $K$ (les mêmes s'appliquent identiquement à $Q$ et $V$).

---

### Étape 1 — Projection unique : `(b, num_tokens, d_in)` → `(b, num_tokens, d_out)`

```python
keys = self.W_key(x)   # (b, num_tokens, d_out)
```

Chaque ligne du tenseur de sortie est le vecteur clé
$\mathbf{k}_i = W_K^\top x_i \in \mathbb{R}^{d_{out}}$ du token $i$.

---

### Étape 2 — Découpe par tête via `.view()` : `(b, num_tokens, d_out)` → `(b, num_tokens, H, d_h)`

```python
keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
```

#### Ce que fait `.view()` concrètement

Pour comprendre `.view()`, il faut d'abord comprendre comment PyTorch stocke les données.

**La mémoire est un couloir de cases numérotées.** Quand PyTorch crée un tenseur, il réserve un bloc de cases consécutives en mémoire (RAM), une case par nombre. Par exemple, une matrice $2 \times 3$

$$M = \begin{pmatrix} a & b & c \\ d & e & f \end{pmatrix}$$

est stockée en mémoire comme une simple file de 6 cases :

$$\boxed{a}\;\boxed{b}\;\boxed{c}\;\boxed{d}\;\boxed{e}\;\boxed{f}$$

PyTorch mémorise juste deux informations séparément : (1) l'adresse de la première case, et (2) la forme `(2, 3)` qui lui permet de calculer où trouver chaque élément — l'élément en ligne $i$, colonne $j$ se trouve à la case $3i + j$.

**`.view()` change la grille de lecture, pas la position physique des données.** Appeler `.view(3, 2)` sur cette même mémoire revient à décider de la lire comme

$$M' = \begin{pmatrix} a & b \\ c & d \\ e & f \end{pmatrix}$$

Ainsi, PyTorch va simplement mettre à jour la forme de `(2, 3)` à `(3, 2)` et recalculer la formule d'accès : l'élément $(i, j)$ est maintenant à la case $2i + j$. C'est tout.

C'est pourquoi on dit que `.view()` est **sans copie** : il n'alloue pas de nouvelle mémoire, ne déplace aucun nombre. Il ne coûte presque rien en temps ni en mémoire, quelle que soit la taille du tenseur.

#### Application à nos clés

**Notation.** Pour un token $i$ et une tête $j$, notons
$\mathbf{k}_{i,j} \in \mathbb{R}^{d_h}$ le sous-vecteur clé associé.

Après l'étape 1, pour un token $i$ fixé, le vecteur
$\mathbf{k}_i \in \mathbb{R}^{d_{out}}$ contient les projections de toutes les têtes
mises bout à bout :

$$K[\,b,\,i,\,:] = \mathbf{k}_i =
\bigl[
  \underbrace{k_{i,1,1},\ldots,k_{i,1,d_h}}_{\displaystyle\mathbf{k}_{i,1}},\;
  \underbrace{k_{i,2,1},\ldots,k_{i,2,d_h}}_{\displaystyle\mathbf{k}_{i,2}},\;
  \ldots,\;
  \underbrace{k_{i,H,1},\ldots,k_{i,H,d_h}}_{\displaystyle\mathbf{k}_{i,H}}
\bigr]$$

`.view(b, num_tokens, H, d_h)` dit à PyTorch : "réinterprète la dimension
$d_{out} = H \cdot d_h$ comme deux dimensions imbriquées". Sans rien déplacer
en mémoire, le même token $i$ est maintenant accessible comme un bloc
$(H, d_h)$ :

$$K[\,b,\,i,\,:,\,:] = \begin{pmatrix} \mathbf{k}_{i,1} \\ \mathbf{k}_{i,2} \\ \vdots \\ \mathbf{k}_{i,H} \end{pmatrix} \in \mathbb{R}^{H \times d_h}$$

La $j$-ième ligne de ce bloc est exactement $\mathbf{k}_{i,j}$, le vecteur clé
du token $i$ pour la tête $j$.

<div align="center">

![view_operation_1](img/view_operation_1.png)
*Figure X : `.view()` réinterprète la dimension $d_{out}$ comme $(H, d_h)$.
Pour chaque token $i$, le vecteur $\mathbf{k}_i \in \mathbb{R}^{d_{out}}$ devient
un bloc $K[b, i, :, :] \in \mathbb{R}^{H \times d_h}$ dont la $j$-ième ligne
est $\mathbf{k}_{i,j}$. Aucune donnée n'est déplacée en mémoire.*

</div>

---

### Étape 3 — Réorganisation par tête via `.transpose(1, 2)` : `(b, num_tokens, H, d_h)` → `(b, H, num_tokens, d_h)`

```python
keys = keys.transpose(1, 2)
```

**Notation.** Notons

$$K_j = \begin{pmatrix} \mathbf{k}_{1,j} \\ \vdots \\ \mathbf{k}_{T,j} \end{pmatrix}
\in \mathbb{R}^{T \times d_h} \qquad (T = \texttt{num\_tokens})$$

la matrice des clés de la tête $j$ sur l'ensemble de la séquence — toutes les
projections du même sous-espace, une ligne par token.

#### Ce que fait `.transpose()` concrètement — et pourquoi c'est différent de `.view()`

Comme `.view()`, `.transpose()` ne déplace rien en mémoire — il change juste la formule d'accès. Mais il crée un **problème** que `.view()` n'avait pas.

Reprenons la matrice $2 \times 3$ :

$$M = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$$

Mémoire physique : $\boxed{1}\;\boxed{2}\;\boxed{3}\;\boxed{4}\;\boxed{5}\;\boxed{6}$

Avec `.view(3, 2)`, PyTorch relisait ces mêmes cases en disant "3 lignes de 2 colonnes" — et chaque ligne consécutive était bien côte à côte en mémoire. Pas de problème.

Soit $$ M.view(3, 2) \Rightarrow \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}$$

Avec `.transpose()`, on obtient logiquement :

$$M^\top = \begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}$$

Pour lire la première ligne de $M^\top$ (les éléments $1$ et $4$), PyTorch doit sauter des cases : il prend la case 0 (1), puis la case 3 (4). Entre eux, il y a les cases 1 et 2 qui appartiennent logiquement à d'autres lignes. L'ordre logique et l'ordre mémoire divergent :

$$\text{Ordre mémoire (intact)} : \boxed{1}\;\boxed{2}\;\boxed{3}\;\boxed{4}\;\boxed{5}\;\boxed{6}$$
$$\text{Ordre logique (après transposition)} : 1\quad 4\quad 2\quad 5\quad 3\quad 6$$

Un tenseur est **non contigu** quand lire ses éléments dans l'ordre logique exige de **sauter des cases** en mémoire.

#### Les deux représentations du tenseur des clés

**Représentation gauche — organisée par token.**
Dans la forme `(b, num_tokens, H, d_h)` issue de `.view()`, la dimension
d'itération principale est le token. Pour un token $i$ fixé, le bloc
$K[\,b,\,i,\,:,\,:]$ regroupe les projections de toutes les têtes pour ce token :

$$K[\,b,\,i,\,:,\,:] = \begin{pmatrix} \mathbf{k}_{i,1} \\ \vdots \\ \mathbf{k}_{i,H} \end{pmatrix} \in \mathbb{R}^{H \times d_h}$$

On lit le tenseur *tête par tête pour un token donné*. En mémoire, les données
sont rangées dans l'ordre : toutes les têtes du token 1, puis toutes les têtes
du token 2, etc. Les matrices $K_1, \ldots, K_H$ existent conceptuellement, mais
leurs lignes sont **entrelacées** — $\mathbf{k}_{1,1}$ suivi de $\mathbf{k}_{1,2}$,
puis $\mathbf{k}_{2,1}$ suivi de $\mathbf{k}_{2,2}$, etc. Pour reconstituer $K_j$,
il faudrait sauter des cases à chaque ligne.

**Représentation droite — organisée par tête.**
Après `.transpose(1, 2)`, la forme est `(b, H, num_tokens, d_h)`. Pour une tête $j$
fixée, le bloc $K[\,b,\,j,\,:,\,:] = K_j$ regroupe les projections de tous les
tokens pour cette tête :

$$K[\,b,\,j,\,:,\,:] = K_j = \begin{pmatrix} \mathbf{k}_{1,j} \\ \vdots \\ \mathbf{k}_{T,j} \end{pmatrix} \in \mathbb{R}^{T \times d_h}$$

On lit le tenseur *token par token pour une tête donnée*. Chaque $K_j$ forme
un bloc logique indépendant. Mais comme vu ci-dessus, `.transpose()` n'a pas
déplacé les données : l'ordre logique et l'ordre mémoire divergent encore. Le
tenseur est non contigu.

La transposition ne change pas les données, seulement la façon dont elles sont
*traversées* : on passe d'une lecture par token à une lecture par tête.

<div align="center">

![view_operation_2](img/view_operation_2.png)
*Figure X : `.transpose(1, 2)` permute les dimensions `num_tokens` et `num_heads`.
À gauche, le bloc $K[b, i, :, :]$ regroupe toutes les têtes d'un token donné.
À droite, le bloc $K[b, j, :, :] = K_j$ regroupe tous les tokens d'une tête donnée.
La mémoire physique ne bouge pas — seul l'ordre de lecture change.*

</div>

---

### Étape 4 — Scores d'attention par multiplication par lots

$Q$, $K$, $V$ ayant tous la forme $(b, H, T, d_h)$, le produit

```python
attn_scores = queries @ keys.transpose(2, 3)
```

calcule $Q_j K_j^\top$ simultanément pour toutes les têtes et tous les éléments du
batch :

$$\texttt{attn\_scores}[\,b,\,j,\,:,\,:] = Q_j K_j^\top \in \mathbb{R}^{T \times T}$$

<div align="center">

![mha_explained_2](img/mha_explained/mha_explained_2.png)
*Figure X : Calcul des scores d'attention par multiplication matricielle par lots
(source : CNRS-FIDLE).*

</div>

---

### Étape 5 — Recombinaison des têtes

Après masquage causal, softmax et dropout, les vecteurs contexte sont recombinés :

```python
context_vec = (attn_weights @ values).transpose(1, 2)
# (b, H, T, d_h) → (b, T, H, d_h)
context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
# (b, T, H, d_h) → (b, T, d_out)
```

#### Ce que fait `.contiguous()` concrètement

`.transpose()` a laissé le tenseur dans un état où ordre logique et ordre mémoire
divergent. `.view()` ne sait pas gérer cet écart — il suppose toujours que lire
les éléments dans l'ordre logique revient à les lire case après case en mémoire.
Appeler `.view()` directement lèverait donc une erreur.

`.contiguous()` résout cela : il alloue un **nouveau bloc mémoire** et y recopie
les données dans l'ordre qui correspond à la forme logique actuelle.

Avec notre exemple numérique : après `.transpose()`, le tenseur est logiquement
$M^\top$ mais la mémoire contient encore $1, 2, 3, 4, 5, 6$ :

$$\text{Ordre mémoire} : \boxed{1}\;\boxed{2}\;\boxed{3}\;\boxed{4}\;\boxed{5}\;\boxed{6}$$
$$\text{Ordre logique} : 1\quad 4\quad 2\quad 5\quad 3\quad 6$$

`.contiguous()` crée un nouveau bloc où les deux ordres coïncident :

$$\text{Nouvelle mémoire} : \boxed{1}\;\boxed{4}\;\boxed{2}\;\boxed{5}\;\boxed{3}\;\boxed{6}$$

`.view()` peut alors s'appliquer sans risque.

> **En résumé.** `.view()` et `.transpose()` sont tous deux sans copie — ils ne
> modifient que la façon dont PyTorch lit la mémoire. Mais `.transpose()` crée un
> écart entre l'ordre logique et l'ordre mémoire que `.view()` ne sait pas gérer.
> `.contiguous()` est la copie explicite qui les réconcilie.

Le `.view(b, num_tokens, self.d_out)` qui suit concatène les sorties des $H$ têtes
pour chaque token :

$$\mathbf{c}_i = \mathbf{c}_{i,1} \| \mathbf{c}_{i,2} \| \cdots \| \mathbf{c}_{i,H}
\in \mathbb{R}^{d_{out}}$$

Enfin, une projection linéaire mélange les informations des différentes têtes :

```python
context_vec = self.out_proj(context_vec)
```

<div align="center">

![mha_explained_3](img/mha_explained/mha_explained_3.png)
*Figure X : Recombinaison des sorties des têtes et projection finale $W_O$
(source : CNRS-FIDLE).*

</div>

## Exercice 3.3 — Initialisation d'un module d'attention de taille GPT-2

### Énoncé

À l'aide de la classe `MultiHeadAttention`, initialiser un module d'attention multi-têtes
possédant le même nombre de têtes que le plus petit modèle GPT-2 (12 têtes). Utiliser
également les dimensions d'embedding d'entrée et de sortie du modèle GPT-2 (768).
Le plus petit GPT-2 supporte une longueur de contexte de 1 024 tokens.

### Solution

```python
import torch
import torch.nn as nn

# Paramètres GPT-2 (smallest variant)
d_in = 768              # input embedding dimension
d_out = 768             # output embedding dimension (same as d_in in GPT-2)
num_heads = 12          # number of attention heads
context_length = 1024   # maximum context window

# Initialiser le module d'attention multi-têtes
mha = MultiHeadAttention(
    d_in=d_in,
    d_out=d_out,
    context_length=context_length,
    dropout=0.1,
    num_heads=num_heads,
    qkv_bias=False
)

# Afficher la configuration
print(f"Configuration d'attention GPT-2 (smallest):")
print(f"  Input dimension (d_in):     {d_in}")
print(f"  Output dimension (d_out):   {d_out}")
print(f"  Number of heads:            {num_heads}")
print(f"  Head dimension:             {d_out // num_heads}")
print(f"  Context length:             {context_length}")
print(f"\nModule: {mha}")
````

---

## Résumé du chapitre 3

Les points clés du chapitre attention et multi-head attention :

**Fondamentaux de l'attention**
- Les mécanismes d'attention transforment un ensemble d'entrées en représentations
  de contexte enrichies incorporant l'information de tous les tokens.
- L'attention simple est une somme pondérée : chaque élément de sortie est une
  combinaison linéaire des entrées, avec des poids appris par le modèle.
- Les poids d'attention sont calculés via des **produits scalaires** entre requêtes
  et clés, une formulation compacte et efficace.

**Mécanisme d'attention à produit scalaire**
- On introduit trois projections linéaires entraînables : **queries** ($Q$), **keys** ($K$),
  et **values** ($V$), permettant au modèle d'apprendre quels types de relations
  chercher et comment les exploiter.
- Le score brut $e_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}}$ est normalisé par la racine
  carrée de la dimension (scaling) pour empêcher l'explosion des variances.
- Le softmax transforme ces scores en poids probabilistes $\alpha_{ij} \in [0, 1]$
  sommant à 1 par ligne.

**Attention causale**
- Pour les modèles de langage lisant et générant de gauche à droite, on applique
  un **masque causal** empêchant chaque token d'accéder aux tokens futurs.
- Deux approches équivalentes : (1) appliquer softmax puis masquer et renormaliser,
  ou (2) remplacer les scores futurs par $-\infty$ avant softmax — la seconde est
  plus efficace en pratique.
- Le dropout appliqué aux poids d'attention renforce la robustesse du modèle en
  le forçant à ne pas sur-dépendre de connexions individuelles.

**Attention multi-têtes**
- Une seule tête d'attention capture une seule façon de combiner les tokens. Plusieurs
  têtes opérant en parallèle permettent au modèle d'explorer plusieurs sous-espaces
  et relations simultanément.
- L'implémentation naïve (stacker des modules mono-têtes) est conceptuellement simple
  mais inefficace : on répète la multiplication matricielle (l'opération la plus coûteuse)
  $H$ fois.
- L'implémentation efficace (classe `MultiHeadAttention`) effectue une seule projection
  de dimension $d_{out} = H \cdot d_h$, puis découpe et réorganise via `.view()` et
  `.transpose()` sans copier les données. Les opérations matricielles résultantes
  (produit par lots sur tous les têtes) exploitent la vectorisation GPU.

**Opérations tensorielles et gestion mémoire**
- `.view()` et `.transpose()` sont des opérations sans copie : elles ne modifient
  que la façon dont PyTorch accède à la mémoire.
- `.transpose()` crée une divergence entre l'ordre logique et l'ordre mémoire, ce
  qui rend `.view()` impossible à appliquer directement après.
- `.contiguous()` résout cet écart en réallouant et réécrivant les données de manière
  que les deux ordres coïncident — une opération nécessaire avant `.view()`.

**Échelle réelle : GPT-2**
- Le plus petit GPT-2 (117M paramètres) dispose de 12 têtes, une dimension d'embedding
  de 768, et une longueur de contexte de 1 024. Chaque tête opère en dimension 64.
- Les architectures modernes de LLM (GPT-3, LLaMA, etc.) augmentent ce nombre de
  têtes (jusqu'à 96 pour GPT-3 large) et les dimensions (jusqu'à 12 288), mais le
  design fondamental reste le même.

## Liens utiles

* **[Understanding buffers](https://github.com/vleonel-junior/Build-a-large-language-model-from-scrach/blob/main/Chapter%203%20Coding%20attention%20mechanism/understanding_buffers.ipynb)** — Exploration des buffers dans les mécanismes d'attention

* **[MHA implementations](https://github.com/vleonel-junior/Build-a-large-language-model-from-scrach/blob/main/Chapter%203%20Coding%20attention%20mechanism/mha_implementations/mha_implementations.ipynb)** — Implémentations pratiques du multi-head attention