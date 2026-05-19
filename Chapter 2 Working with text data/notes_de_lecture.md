# Notes de lecture : Build a Large Language Model (from scratch)

## Chapitre 2 : Working with text data

### Introduction : Pipeline de prÃ©paration des donnÃ©es

Avant mÃªme de songer Ã  l'architecture ou Ã  l'entraÃ®nement d'un LLM, l'Ã©tape la plus cruciale est la prÃ©paration des donnÃ©es d'entraÃ®nement. C'est l'objectif de la toute premiÃ¨re Ã©tape de la phase 1 du dÃ©veloppement d'un modÃ¨le de langage.

Pour pouvoir traiter du texte, un LLM ne lit pas des phrases brutes, mais convertit le texte suivant un pipeline bien prÃ©cis :
1. **La tokenisation :** Le texte brut est dÃ©coupÃ© en petites unitÃ©s appelÃ©es *tokens* (qui peuvent Ãªtre des mots entiers, ou des sous-mots). Des mÃ©thodes avancÃ©es comme le *Byte Pair Encoding* (BPE) sont gÃ©nÃ©ralement utilisÃ©es sur les modÃ¨les rÃ©cents comme GPT.
2. **L'Ã©chantillonnage :** GrÃ¢ce Ã  une approche par Â« fenÃªtre glissante Â» (sliding window), on extrait des paires d'entrÃ©es et de sorties permettant d'entraÃ®ner le modÃ¨le Ã  la tÃ¢che de prÃ©diction du mot suivant.
3. **La vectorisation :** Les tokens extraits sont ensuite convertis en vecteurs de nombres (embeddings) que le rÃ©seau de neurones va pouvoir ingÃ©rer et traiter mathÃ©matiquement.

<div align="center">

![Figure 2.1](img/figure_2.1.png)


*Figure 2.1 : Les trois grandes Ã©tapes pour coder un LLM. Ce chapitre se concentre sur l'Ã©tape 1 de la phase 1 : l'implÃ©mentation du pipeline d'Ã©chantillonnage de donnÃ©es.*

</div>

<br>

### 2.1 Comprendre les embeddings de mots

Les rÃ©seaux de neurones ne peuvent pas traiter directement le texte brut, car les algorithmes nÃ©cessitent des valeurs numÃ©riques continues pour fonctionner. La mÃ©thode pour accomplir cela s'appelle l'**embedding** (ou plongement vectoriel).

Un embedding consiste Ã  projeter des objets discrets (mots, phrases, audios, images) vers des points dans un format que la machine peut traiter : un espace vectoriel continu (une suite de nombres). Il est essentiel de comprendre qu'il existe un type de modÃ¨le d'embedding pour chaque type de format de donnÃ©es (on ne peut pas utiliser un modÃ¨le textuel sur de la vidÃ©o).

<div align="center">

![Figure 2.2](img/figure_2.2.png)


*Figure 2.2 : Les modÃ¨les de deep learning ne peuvent pas traiter des formats de donnÃ©es tels que la vidÃ©o, l'audio et le texte dans leur forme brute. Ainsi, nous utilisons un modÃ¨le d'embedding pour transformer ces donnÃ©es brutes en une reprÃ©sentation vectorielle dense que les architectures de deep learning peuvent facilement comprendre et traiter. Plus prÃ©cisÃ©ment, cette figure illustre le processus de conversion de donnÃ©es brutes en un vecteur numÃ©rique tridimensionnel.*

</div>

Dans ce livre, nous nous intÃ©ressons uniquement aux **embeddings de mots** puisque la gÃ©nÃ©ration des LLM opÃ¨re un mot Ã  la fois. Historiquement, on utilisait des modÃ¨les tiers comme **Word2Vec** pour transformer chaque mot en vecteur. La logique mathÃ©matique est Ã©lÃ©gante : les mots partageant des contextes similaires acquiÃ¨rent des valeurs mathÃ©matiques similaires et se retrouvent par consÃ©quent regroupÃ©s (clustered) gÃ©omÃ©triquement lorsqu'on les affiche dans un espace en deux dimensions. 

<div align="center">

![Figure 2.3](img/figure_2.3.png)


*Figure 2.3 : Si les embeddings de mots sont bidimensionnels, nous pouvons les tracer dans un nuage de points en 2D pour les visualiser, comme montrÃ© ici. Lors de l'utilisation de techniques d'embedding de mots, telles que Word2Vec, les mots correspondant Ã  des concepts similaires apparaissent souvent proches les uns des autres dans l'espace d'embedding. Par exemple, diffÃ©rents types d'oiseaux apparaissent plus proches les uns des autres dans l'espace d'embedding qu'ils ne le sont des pays et des villes.*

</div>

Cependant, les LLM modernes ne s'appuient pas sur Word2Vec. Ils utilisent leur propre **couche d'embedding intÃ©grÃ©e**, qui s'entraÃ®ne et s'optimise en mÃªme temps que le reste du modÃ¨le, permettant d'adapter les vecteurs aux spÃ©cificitÃ©s exactes des donnÃ©es cibles.

Enfin, la **dimensionnalitÃ©** de ces vecteurs est clÃ© : un espace Ã  deux dimensions est utile pour enseigner et visualiser (comme la figure 2.3), mais dans un modÃ¨le rÃ©el, le nombre de dimensions chiffre trÃ¨s vite pour capturer toute l'expressivitÃ© et la nuance de la langue. Plus il y a de dimensions, plus la prÃ©cision augmente, au dÃ©triment de l'efficacitÃ© calculatoire. Un petit GPT-2 utilise 768 dimensions pour un seul mot, alors que l'immense GPT-3 en requiert 12 288 !

## 2.2 La tokenization du texte (Tokenizing text)

La tokenization (ou segmentation en lexÃ¨mes) est une Ã©tape de prÃ©traitement indispensable avant la crÃ©ation d'embeddings pour un LLM. Elle consiste Ã  diviser le texte d'entrÃ©e en *tokens* individuels, qui peuvent Ãªtre des mots isolÃ©s ou des caractÃ¨res spÃ©ciaux, y compris la ponctuation.

<div align="center">

![Figure 2.4](img/figure_2.4.png)


*Figure 2.4 : Une vue des Ã©tapes de traitement du texte dans le contexte d'un LLM. Ici, nous divisons un texte d'entrÃ©e en tokens individuels (mots ou caractÃ¨res spÃ©ciaux).*

</div>

Dans cette section, nous utiliserons la courte nouvelle *"The Verdict"* d'Edith Wharton, disponible dans le domaine public.

Vous pouvez tÃ©lÃ©charger et lire ce texte avec le code Python suivant :

```python
import urllib.request
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])
```

**RÃ©sultat :**
```text
Total number of character: 20479
I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow
enough--so it was no
```

Bien que l'entraÃ®nement de vrais LLMs implique souvent des millions d'articles (des gigaoctets de texte), nous utiliserons cet Ã©chantillon de 20 479 caractÃ¨res Ã  des fins Ã©ducatives pour pouvoir exÃ©cuter le code en temps raisonnable sur du matÃ©riel grand public.

### CrÃ©ation d'un tokenizer de base avec Python

Afin de diviser le texte et d'obtenir une liste de tokens, nous faisons une courte incursion dans la bibliothÃ¨que d'expressions rÃ©guliÃ¨res de Python (`re`).

Nous Ã©vitons de convertir tout le texte en minuscules, car la capitalisation aide les LLMs Ã  :
- Distinguer les noms propres des noms communs.
- Comprendre la structure des phrases.
- Apprendre Ã  gÃ©nÃ©rer du texte avec une capitalisation correcte.

Voici une premiÃ¨re tentative de sÃ©paration basÃ©e uniquement sur les espaces :

```python
import re
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
print(result)
```

**RÃ©sultat :**
```text
['Hello,', ' ', 'world.', ' ', 'This,', ' ', 'is', ' ', 'a', ' ', 'test.']
```

Ce schÃ©ma fonctionne en grande partie, mais la ponctuation reste collÃ©e aux mots (`"Hello,"`). Pour corriger cela, sÃ©parons Ã©galement sur les virgules et les points (`r'([,.]|\s)'`) :

```python
result = re.split(r'([,.]|\s)', text)
print(result)
```

**RÃ©sultat :**
```text
['Hello', ',', '', ' ', 'world', '.', '', ' ', 'This', ',', '', ' ', 'is', ' ', 'a', ' ', 'test', '.', '']
```

Un petit problÃ¨me subsiste : la liste inclut encore les espaces et des chaÃ®nes vides. Nous pouvons supprimer ces caractÃ¨res superflus avec `.strip()` :

```python
result = [item for item in result if item.strip()]
print(result)
```

**RÃ©sultat :**
```text
['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.']
```

> **Note sur les espaces :**
> Lors du dÃ©veloppement d'un tokenizer simple, le fait d'encoder les espaces comme des caractÃ¨res sÃ©parÃ©s ou de les supprimer (via `.strip()`) dÃ©pend de votre application. Leurs suppressions rÃ©duisent les besoins en mÃ©moire et calcul. Cependant, conserver les espaces est utile pour les modÃ¨les sensibles Ã  la structure exacte du texte (comme le code Python, sensible Ã  l'indentation). Ici, nous les supprimons pour simplifier.

Complexifions l'expression rÃ©guliÃ¨re pour gÃ©rer d'autres signes de ponctuation et les double-tirets, similaires Ã  ceux rencontrÃ©s dans *"The Verdict"* :

```python
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)
```

**RÃ©sultat :**
```text
['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
```

<div align="center">

![Figure 2.5](img/figure_2.5.png)


*Figure 2.5 : Le schÃ©ma de tokenization sÃ©pare correctement le texte en mots individuels et ponctuations.*

</div>

Maintenant que nous avons un tokenizer de base fonctionnel, appliquons-le Ã  l'entiÃ¨retÃ© de la nouvelle d'Edith Wharton :

```python
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

print(len(preprocessed))
print(preprocessed[:30])
```

**RÃ©sultat :**
```text
4690
['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 'genius', '--', 'though', 'a', 'good', 'fellow', 'enough', '--', 'so', 'it', 'was', 'no', 'great', 'surprise', 'to', 'me', 'to', 'hear', 'that', ',', 'in']
```


## 2.3 Convertir les tokens en identifiants (Token IDs)

Une fois le texte divisÃ© en tokens (chaÃ®nes de caractÃ¨res), l'Ã©tape suivante consiste Ã  les convertir en nombres entiers (*Token IDs*). C'est une Ã©tape intermÃ©diaire obligatoire avant de gÃ©nÃ©rer les vecteurs d'embeddings.

Pour transformer les tokens en identifiants, nous devons d'abord construire un **vocabulaire**. Ce vocabulaire mappe chaque mot et caractÃ¨re spÃ©cial unique Ã  un nombre entier de faÃ§on unique.

<div align="center">

![Figure 2.6](img/figure_2.6.png)

*Figure 2.6 : Construction d'un vocabulaire en tokenisant l'ensemble du dataset d'entraÃ®nement. Les tokens sont extraits, triÃ©s par ordre alphabÃ©tique, et les doublons retirÃ©s. Le vocabulaire fait correspondre chaque token unique Ã  une valeur entiÃ¨re unique.*

</div>

CrÃ©ons la liste de tous les tokens uniques et trions-les pour dÃ©terminer la taille de notre vocabulaire :

```python
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)
```

**RÃ©sultat :**
```text
1130
```

Le vocabulaire contient 1 130 tokens diffÃ©rents. Nous pouvons maintenant crÃ©er le dictionnaire qui associe chaque token Ã  un numÃ©ro :

```python
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break
```

**RÃ©sultat :**
```text
('!', 0)
('"', 1)
("'", 2)
...
('Her', 49)
('Hermia', 50)
```

<div align="center">

![Figure 2.7](img/figure_2.7.png)

*Figure 2.7 : Ã€ partir d'un nouvel Ã©chantillon de texte, nous le tokenisons et utilisons le vocabulaire pour convertir les tokens textuels en token IDs.*

</div>

### ImplÃ©mentation d'une classe Tokenizer

Pour automatiser ce processus, nous allons implÃ©menter une classe Python `SimpleTokenizerV1`. Elle comprendra :
- Une mÃ©thode `encode()` pour diviser un texte en tokens et le transformer en IDs.
- Une mÃ©thode `decode()` pour effectuer l'opÃ©ration inverse (de Token IDs vers textes), indispensable pour lire la sortie gÃ©nÃ©rÃ©e par le modÃ¨le.

```python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab  # Stocke le vocabulaire pour l'encodage
        self.int_to_str = {i:s for s,i in vocab.items()}  # Vocabulaire inversÃ© pour le dÃ©codage

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Supprime les espaces insÃ©rÃ©s avant la ponctuation
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) 
        return text
```

<div align="center">

![Figure 2.8](img/figure_2.8.png)

*Figure 2.8 : Les implÃ©mentations de tokenizers partagent deux mÃ©thodes communes : encode (convertit le texte en ID via le vocabulaire) et decode (reconvertit les IDs en texte naturel).*

</div>

Testons notre classe sur un extrait de la nouvelle :

```python
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
```

**RÃ©sultat :**
```text
[1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]
```

DÃ©codons maintenant cette liste d'IDs pour vÃ©rifier si nous retrouvons la phrase originale :

```python
print(tokenizer.decode(ids))
```

**RÃ©sultat :**
```text
'" It\' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
```

Le dÃ©codeur fonctionne bien ! Essayons maintenant un nouveau texte qui n'est pas issu de la nouvelle d'Edith Wharton :

```python
text = "Hello, do you like tea?"
print(tokenizer.encode(text))
```

**RÃ©sultat :**
```text
KeyError: 'Hello'
```

**ProblÃ¨me :** Le mot "Hello" n'apparaÃ®t pas dans la nouvelle *"The Verdict"*, il est donc absent de notre vocabulaire. Cela dÃ©montre pourquoi les LLMs doivent Ãªtre entraÃ®nÃ©s sur de gigantesques ensembles de donnÃ©es diversifiÃ©s pour Ã©tendre leur vocabulaire (et pourquoi nous aurons besoin de tokens spÃ©ciaux pour gÃ©rer les mots inconnus).

## 2.4 Ajouter des tokens de contexte spÃ©ciaux (Adding special context tokens)

Il est indispensable de modifier le tokenizer pour gÃ©rer les mots inconnus. De plus, l'ajout de tokens de contexte spÃ©ciaux permet d'amÃ©liorer la comprÃ©hension du modÃ¨le, par exemple pour marquer la fin ou le dÃ©but d'un document. Nous allons ajouter deux nouveaux tokens : `<|unk|>` pour les mots inconnus, et `<|endoftext|>` pour sÃ©parer des documents textuels indÃ©pendants.

<div align="center">

![Figure 2.9](img/figure_2.9.png)

*Figure 2.9 : Ajout des tokens spÃ©ciaux `<|unk|>` (pour les mots inconnus) et `<|endoftext|>` (pour sÃ©parer deux sources de texte non liÃ©es) au vocabulaire.*

</div>

L'ajout du token `<|endoftext|>` est crucial lorsqu'on entraÃ®ne des LLMs de type GPT sur de multiples documents ou livres indÃ©pendants. Cela aide le modÃ¨le Ã  comprendre que, bien que concatÃ©nÃ©s Ã  la chaÃ®ne pour l'entraÃ®nement, ces textes n'ont aucun lien contextuel entre eux.

<div align="center">

![Figure 2.10](img/figure_2.10.png)

*Figure 2.10 : Lors du traitement de plusieurs sources de texte indÃ©pendantes, le token `<|endoftext|>` agit comme un marqueur signalant le dÃ©but ou la fin d'un segment.*

</div>

### Mise Ã  jour du vocabulaire

Ajoutons ces deux tokens spÃ©ciaux Ã  la suite de notre liste de mots uniques, puis vÃ©rifions la nouvelle taille du vocabulaire :

```python
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items()))
```

**RÃ©sultat :**
```text
1132
```
Le vocabulaire compte bien Ã  prÃ©sent 1 132 valeurs (au lieu de 1 130). Imprimons les cinq derniÃ¨res entrÃ©es du dictionnaire pour le confirmer :

```python
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)
```

**RÃ©sultat :**
```text
('younger', 1127)
('your', 1128)
('yourself', 1129)
('<|endoftext|>', 1130)
('<|unk|>', 1131)
```

### Le Tokenizer V2 gÃ©rant les mots inconnus

Nous pouvons ajuster la mÃ©thode `encode` de notre classe prÃ©cÃ©dente. DÃ©sormais, si un mot rencontrÃ© dans le texte fourni n'est pas prÃ©sent dans la base de donnÃ©es de notre `self.str_to_int`, nous lui associons d'office le token `<|unk|>`.

```python
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        
        # Filtre de sÃ©curitÃ© pour les mots inconnus
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
```

Testons cette nouvelle version sur un Ã©chantillon composÃ© de deux phrases indÃ©pendantes concatÃ©nÃ©es avec notre fameux marqueur, pour vÃ©rifier la mise en place du texte de test :

```python
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)
```

**RÃ©sultat :**
```text
Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.
```

Maintenant, encodons ce texte complet et regardons les identifiants gÃ©nÃ©rÃ©s :

```python
tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))
```

**RÃ©sultat :**
```text
[1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]
```
*(On observe bien la prÃ©sence du token *1130* correspondant au `<|endoftext|>` et deux compteurs *1131* pour `<|unk|>` correspondant aux mots hors-vocabulaire).*

DÃ©cryptons les identitÃ©s pour le voir de nos propres yeux :

```python
print(tokenizer.decode(tokenizer.encode(text)))
```

**RÃ©sultat :**
```text
<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.
```

En comparant ce texte dÃ©tokenisÃ© avec le texte d'entrÃ©e original, nous pouvons en dÃ©duire que le dataset d'entraÃ®nement, l'histoire courte d'Edith Wharton "The Verdict", ne contient pas les mots "Hello" et "palace". Ces mots ont donc Ã©tÃ© remplacÃ©s par `<|unk|>`.

### Autres types de tokens contextuels

Selon le LLM, certains chercheurs considÃ¨rent Ã©galement d'autres tokens spÃ©ciaux supplÃ©mentaires tels que :
- **`[BOS]` (Beginning of sequence)** : Ce token marque le dÃ©but d'un texte. Il indique au LLM oÃ¹ commence un fragment de contenu.
- **`[EOS]` (End of sequence)** : Ce token est positionnÃ© Ã  la fin d'un texte et est particuliÃ¨rement utile lorsque l'on concatÃ¨ne plusieurs textes non liÃ©s, de maniÃ¨re similaire Ã  `<|endoftext|>`. Par exemple, lors de la combinaison de deux articles WikipÃ©dia ou livres diffÃ©rents, le token `[EOS]` indique oÃ¹ l'un se termine et oÃ¹ le suivant commence.
- **`[PAD]` (Padding)** : Lors de l'entraÃ®nement de LLMs avec des tailles de lots (*batch sizes*) supÃ©rieures Ã  un, le batch peut contenir des textes de longueurs variables. Pour s'assurer que tous les textes ont la mÃªme longueur, les textes plus courts sont prolongÃ©s ou "rembourrÃ©s" (*padded*) en utilisant le token `[PAD]`, jusqu'Ã  atteindre la longueur du texte le plus long du lot.

> **La particularitÃ© du tokenizer des modÃ¨les GPT**
>
> Le tokenizer des modÃ¨les GPT se distingue par sa simplicitÃ©. Au lieu de multiplier les tokens spÃ©ciaux, il n'utilise que `<|endoftext|>` comme Ã©quivalent Ã  `[BOS]` et `[EOS]`.
>
> Ce token `<|endoftext|>` sert Ã©galement pour le *padding*. Lors de l'entraÃ®nement par lots (*batchs*), un masque (*mask*) est appliquÃ© pour que le modÃ¨le ignore simplement ces tokens de remplissage. Le choix du token spÃ©cifique pour le padding n'a donc aucune importance.
>
> Enfin, ce tokenizer n'utilise pas de token `<|unk|>` pour les mots hors-vocabulaire (*out-of-vocabulary*). Il utilise plutÃ´t un algorithme appelÃ© encodage par paire d'octets (*Byte Pair Encoding* ou `BPE`), qui dÃ©compose les mots inconnus en sous-mots (*subword units*), comme nous le verrons dans la section suivante.

## 2.5 L'Encodage par paire d'octets (Byte Pair Encoding - BPE)

Examinons un schÃ©ma de tokenisation plus sophistiquÃ© basÃ© sur un concept appelÃ© encodage par paire d'octets (Byte Pair Encoding - BPE). Le tokenizer BPE a Ã©tÃ© utilisÃ© pour entraÃ®ner des LLMs tels que GPT-2, GPT-3 et le modÃ¨le original utilisÃ© dans ChatGPT.

Ici, nous allons utiliser une bibliothÃ¨que open-source Python existante appelÃ©e `tiktoken` (https://github.com/openai/tiktoken), qui implÃ©mente l'algorithme BPE de maniÃ¨re trÃ¨s. De mÃªme que pour d'autres bibliothÃ¨ques Python, nous pouvons installer la bibliothÃ¨que `tiktoken` via l'installateur de paquets `pip` depuis le terminal :

```bash
pip install tiktoken
```

VÃ©rification de la version :

```python
from importlib.metadata import version
import tiktoken

print("tiktoken version:", version("tiktoken"))
```

**RÃ©sultat :**
```text
tiktoken version: 0.7.0
```

Initialisation d'un tokenizer BPE (modÃ¨le "gpt2") :

```python
tokenizer = tiktoken.get_encoding("gpt2")
```

**Encodage d'un texte (y compris les tokens de contrÃ´le contextuel) :**

```python
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)
# Le paramÃ¨tre allowed_special Ã©vite que le tokenizer lÃ¨ve une erreur en voyant le token <|endoftext|>
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
```

**RÃ©sultat :**
```text
[15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 617, 34680, 27271, 13]
```

**DÃ©codage inverse :**

```python
strings = tokenizer.decode(integers)
print(strings)
```

**RÃ©sultat :**
```text
Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.
```

### Observations fondamentales sur le Tokenizer BPE

Cette expÃ©rimentation met en Ã©vidence **deux faits remarquables** inhÃ©rents au tokenizer BPE de la famille GPT :

1. **La position du token `<|endoftext|>`** : Son ID attribuÃ© est trÃ¨s grand (50256). C'est logique : le vocabulaire total des modÃ¨les type GPT-2 ou GPT-3 est limitÃ© Ã  **50 257** tokens, avec `<|endoftext|>` occupant l'ultime position Ã  la fin.
2. **La gestion robuste des mots "hors-vocabulaire" (OOV)** : Le tokenizer parvient Ã  re-transformer et dÃ©coder parfaitement `someunknownPlace` sans avoir Ã  effectuer de "fallback" vers un token aveugle comme `<|unk|>`.

**Comment le BPE se passe-t-il totalement du token `<|unk|>` ?**

La force du BPE rÃ©side dans sa capacitÃ© Ã  **fractionner** un mot totalement inconnu. Au lieu de lever une erreur ou de le remplacer brutalement par `<|unk|>`, le BPE segmente le mot en fragments plus petits : des syllabes / sous-mots ("subwords") connus ou, s'il le faut en dernier recours, en caractÃ¨res ou octets individuels.

<div align="center">

> ![Figure 2.11](img/figure_2.11.png)
>
> *Figure 2.11 : Les tokenizers BPE dÃ©composent les mots inconnus en sous-mots et en caractÃ¨res individuels. Ainsi, un tokenizer BPE peut analyser n'importe quel mot et n'a pas besoin de remplacer les mots inconnus par des tokens spÃ©ciaux, tels que `<|unk|>`.*

</div>

Cette capacitÃ© de dÃ©composer les mots inconnus en caractÃ¨res individuels garantit que le tokenizer, et par consÃ©quent le LLM, peut traiter n'importe quel texte, mÃªme s'il contient des mots absents de ses donnÃ©es d'entraÃ®nement

### Exercice 2.1 : Encodage par paire d'octets de mots inconnus

> Essayez le tokenizer BPE de la bibliothÃ¨que `tiktoken` sur le mot inconnu "Akwirw ier" et affichez les identifiants de tokens (token IDs) individuels. Ensuite, appelez la fonction de dÃ©codage sur chacun des entiers obtenus dans cette liste pour reproduire le mappage logiquement dictÃ© par le BPE. Pour finir, appelez la mÃ©thode de dÃ©codage sur l'ensemble final des identifiants de tokens pour vÃ©rifier s'il parvient Ã  reconstruire l'entrÃ©e d'origine de la figure 2.11.

**Solution :**

En passant manuellement chaque sous-fragment perÃ§u du mot de la figure 2.11 au tokenizer :

```python
print(tokenizer.encode("Ak"))
print(tokenizer.encode("w"))
# ...
```

**RÃ©sultat :**
```text
[33901]
[86]
# ...
```

Une fois assemblÃ©s, un unique appel repasse la liste d'IDs au tokenizer, reconstituant fidÃ¨lement la chaÃ®ne de dÃ©part :

```python
print(tokenizer.decode([33901, 86, 343, 86, 220, 959]))
```

**RÃ©sultat :**
```text
Akwirw ier
```



En bref, le BPE construit son vocabulaire de maniÃ¨re progressive, en partant de la plus petite unitÃ©. Il initialise d'abord son vocabulaire avec tous les caractÃ¨res individuels (par exemple "a", "b", etc.). Ensuite, il repÃ¨re les caractÃ¨res qui apparaissent le plus souvent cÃ´te Ã  cÃ´te pour les fusionner en sous-mots. Par exemple, si "d" et "e" sont trÃ¨s souvent adjacents, ils sont fusionnÃ©s pour crÃ©er le sous-mot "de" (trÃ¨s courant dans des mots comme "define" ou "made"). Ce mÃ©canisme se rÃ©pÃ¨te itÃ©rativement, fusionnant les sous-mots les plus frÃ©quents en mots entiers, uniquement sur la base de leur frÃ©quence d'apparition.

## 2.6 Ã‰chantillonnage des donnÃ©es avec une fenÃªtre glissante (Sliding window)

Pour entraÃ®ner un LLM, on gÃ©nÃ¨re des paires entrÃ©e-cible (input-target pairs). La tÃ¢che du modÃ¨le Ã©tant de prÃ©dire le mot suivant, la sÃ©quence cible (`y`) correspond exactement Ã  la sÃ©quence d'entrÃ©e (`x`), mais dÃ©calÃ©e d'une position vers la droite.

<div align="center">

> ![Figure 2.12](img/figure_2.12.png)
>
> *Figure 2.12 : Extraction de blocs d'entrÃ©e (input blocks) Ã  partir d'un Ã©chantillon de texte pour l'entraÃ®nement du LLM. La tÃ¢che consiste Ã  prÃ©dire le mot suivant le bloc d'entrÃ©e, en masquant les mots qui suivent la cible. (La tokenisation est omise ici pour plus de clartÃ©).*

</div>

Pour crÃ©er ces paires, on fait glisser une fenÃªtre sur le texte tokenisÃ©. La taille du contexte (`context_size` ou `max_length`) dÃ©termine le nombre de tokens de l'entrÃ©e.

```python
# Exemple de dÃ©calage d'une position pour crÃ©er la cible
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}") # [290, 4920, 2241, 287]
print(f"y: {y}") # [4920, 2241, 287, 257]
```

### ImplÃ©mentation du pipeline de donnÃ©es avec PyTorch

Pour l'entraÃ®nement, les donnÃ©es doivent Ãªtre converties en tenseurs et organisÃ©es en lots. On utilise pour cela deux classes standard de PyTorch : `Dataset` et `DataLoader`.

<div align="center">

> ![Figure 2.13](img/figure_2.13.png)
>
> *Figure 2.13 : Pour une efficacitÃ© maximale, les entrÃ©es sont regroupÃ©es dans un tenseur `x`, oÃ¹ chaque ligne reprÃ©sente le contexte d'entrÃ©e. Un second tenseur `y` contient les cibles prÃ©dictives correspondantes (les mots suivants), crÃ©Ã©es en dÃ©calant l'entrÃ©e d'une position.*

</div>

**1. La classe Dataset (`GPTDatasetV1`)** : 
Cette classe dÃ©finit comment dÃ©couper le texte en sÃ©quences individuelles. Elle divise le texte en blocs de la taille de `max_length` pour les entrÃ©es, et crÃ©e les blocs cibles correspondants en les dÃ©calant d'un token.

```python
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        # Tokenisation de tout le texte
        token_ids = tokenizer.encode(txt)
        
        # Utilisation d'une fenÃªtre glissante pour crÃ©er les sÃ©quences
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```

**2. La classe DataLoader** : 
Le `DataLoader` regroupe les sÃ©quences du `Dataset` en lots (batches). Cela permet au modÃ¨le de traiter plusieurs exemples en parallÃ¨le.

```python
import tiktoken

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader
```

### ParamÃ¨tres importants du DataLoader

- **Batch Size (Taille du lot)** : Le nombre de sÃ©quences traitÃ©es simultanÃ©ment. Une taille de 1 est utile pour l'illustration, mais en apprentissage profond, on utilise des lots plus grands pour stabiliser les mises Ã  jour du modÃ¨le.
- **Drop Last (`drop_last=True`)** : Si le nombre total de sÃ©quences n'est pas divisible par la taille du lot, le dernier lot sera incomplet. L'ignorer (le supprimer) permet d'Ã©viter des instabilitÃ©s (pics de perte) de l'entraÃ®nement.
- **Stride (Le pas)** : DÃ©termine de combien de positions la fenÃªtre glissante avance pour extraire la sÃ©quence suivante. 
  - Un `stride` de 1 fait avancer la fenÃªtre d'un seul token, crÃ©ant beaucoup de chevauchements entre les sÃ©quences consÃ©cutives (pratique pour visualiser le mÃ©canisme).
  - En pratique lors de l'entraÃ®nement, on dÃ©finit souvent le `stride` Ã  la mÃªme valeur que `max_length`. Cela empÃªche les sÃ©quences de se chevaucher, limitant ainsi le surapprentissage (overfitting).

<div align="center">

> ![Figure 2.14](img/figure_2.14.png)
>
> *Figure 2.14 : En dÃ©finissant un pas (stride) Ã©gal Ã  la taille de la fenÃªtre d'entrÃ©e (input window size), on Ã©vite tout chevauchement entre les lots.*

</div>

### Exercice 2.2 : Data loaders avec diffÃ©rents strides et context sizes

> Essayez le data loader avec d'autres paramÃ¨tres comme `max_length=2` et `stride=2`, ou encore `max_length=8` et `stride=2` pour dÃ©velopper votre intuition de la mÃ©canique de la fenÃªtre glissante.


## 2.7 Création des embeddings de tokens (Token embeddings)

La dernière étape de la préparation des données consiste à convertir les identifiants de tokens (token IDs) en vecteurs d'intégration continus (embedding vectors). Cette représentation vectorielle est requise car les LLMs sont des réseaux de neurones profonds entraînés par l'algorithme de rétropropagation (backpropagation).

Les poids de la matrice d'embeddings sont initialisés avec de petites valeurs aléatoires qui seront optimisées pendant l'entraînement du modèle.

<div align="center">

> ![Figure 2.15](img/figure_2.15.png)
>
> *Figure 2.15 : La préparation du texte passe par la tokenisation, la conversion en identifiants (token IDs), et enfin la projection de ces identifiants en vecteurs continus via une couche d'embedding.*

</div>

### Fonctionnement d'une couche d'embedding avec PyTorch

On instancie une telle couche avec `torch.nn.Embedding`. Ses dimensions dépendent de deux paramètres :
- `vocab_size` : La taille du vocabulaire (le nombre de lignes).
- `output_dim` : Le nombre de dimensions de chaque vecteur d'embedding (le nombre de colonnes).

L'application d'un token ID à cette couche effectue une **opération de recherche (lookup operation)** : elle récupère directement la ligne de la matrice correspondant à cet identifiant.

```python
# Exemple d'extraction de vecteurs pour input_ids = [2, 3, 5, 1]
# vocab_size = 6, output_dim = 3
print(embedding_layer(input_ids))

tensor([[ 1.2753, -0.2010, -0.1606],   # Ligne d'index 2
        [-0.4015,  0.9666, -1.1481],   # Ligne d'index 3
        [-2.8400, -0.7849, -1.4096],   # Ligne d'index 5
        [ 0.9178,  1.5810,  1.3010]],  # Ligne d'index 1
       grad_fn=<EmbeddingBackward0>)
```

<div align="center">

> ![Figure 2.16](img/figure_2.16.png)
>
> *Figure 2.16 : L'extraction de vecteurs d'embedding. Chaque token ID sert d'index pour extraire la ligne correspondante depuis la matrice de poids de la couche d'embedding.*

</div>

> **Note sur le One-hot encoding :**
> Utiliser une couche d'embedding est mathématiquement et fondamentalement équivalent à appliquer un encodage "one-hot" suivi d'une multiplication matricielle (couche fully connected). La couche d'embedding est cependant une implémentation beaucoup plus efficace en termes de calculs, tout en restant un composant différentiable pour la rétropropagation.
