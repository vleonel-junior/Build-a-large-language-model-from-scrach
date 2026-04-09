# Notes de lecture : Build a Large Language Model (from scratch)

## Chapitre 1 : Understanding large language models

### 1.1 Qu'est-ce qu'un LLM ?

Un LLM (Large Language Model, ou grand modèle de langage) est un réseau de neurones profond conçu pour comprendre, générer et répondre à du texte de manière humaine. Le terme « large » fait référence à la fois à la taille du modèle (qui possède des dizaines, voire des centaines de milliards de paramètres ou poids ajustables optimisés pendant l'entraînement) et à l'immense ensemble de données textuelles sur lequel il est entraîné, couvrant parfois une grande partie de tout le texte disponible publiquement sur Internet.

L'apprentissage de base d'un LLM repose sur la prédiction du mot suivant. Cette tâche, bien qu'en apparence très simple, exploite la nature séquentielle du langage pour entraîner les modèles à comprendre le contexte, la structure et les relations dans le texte, produisant ainsi des modèles extrêmement performants. 

Pour accomplir cela, les LLM utilisent une architecture appelée « transformer », qui inclut un mécanisme d'attention sélective leur permettant de se concentrer sur différentes parties de l'entrée lors des prédictions, ce qui les rend particulièrement aptes à gérer les nuances et la complexité du langage humain.

Puisque les LLM sont capables de générer du texte, ils sont souvent considérés comme une forme d'intelligence artificielle générative (Generative AI ou GenAI). Ils représentent une application spécifique des techniques de *deep learning*, qui est lui-même un sous-domaine du *machine learning*, tous deux appartenant au vaste domaine de l'Intelligence Artificielle (IA). L'IA englobe la création de machines capables d'effectuer des tâches nécessitant une intelligence humaine.

<div align="center">

![Figure 1.1](img/figure_1.1.png)

</div>
*Figure 1.1 : Comme le suggère cette représentation hiérarchique de la relation entre les différents domaines, les LLM représentent une application spécifique des techniques de deep learning, utilisant leur capacité à traiter et générer du texte de type humain. Le deep learning est une branche spécialisée du machine learning qui se concentre sur l’utilisation de réseaux de neurones à plusieurs couches. Le machine learning et le deep learning visent à mettre en œuvre des algorithmes permettant aux ordinateurs d’apprendre à partir de données et d’effectuer des tâches qui nécessitent généralement une intelligence humaine.*

\
La distinction fondamentale entre le *machine learning* traditionnel et le *deep learning* réside dans l'extraction des caractéristiques. Le *machine learning* consiste à développer des algorithmes capables d'apprendre à partir de données sans être explicitement programmés, mais il nécessite une extraction manuelle des caractéristiques. Dans un filtre anti-spam classique par exemple, des experts humains doivent identifier manuellement les caractéristiques pertinentes (comme la fréquence du mot « gratuit » ou la présence de liens suspects). 

À l'inverse, le *deep learning* utilise des réseaux de neurones comportant trois couches ou plus (réseaux de neurones profonds) pour modéliser directement des motifs et des abstractions complexes dans les données, sans nécessiter d'extraction manuelle des caractéristiques par des experts humains. (À noter que, dans un cadre d'apprentissage supervisé comme un filtre anti-spam, les deux méthodes nécessitent tout de même la collecte de labels initiaux).

### 1.2 Applications des LLM

Grâce à leur capacité à analyser des données textuelles non structurées, les LLM trouvent leur utilité dans une multitude de domaines, allant de la traduction automatique à la synthèse de texte en passant par l'analyse de sentiment. Ils sont aujourd'hui activement utilisés pour créer du contenu original tel que de la fiction, des articles de blog ou du code informatique.

Au-delà de la simple génération, les LLM constituent le moteur d'assistants virtuels et de chatbots sophistiqués (comme ChatGPT ou Gemini). Ces outils transforment notre manière d'interagir avec la technologie, en rendant les systèmes plus conversationnels et en améliorant significativement l'interaction avec les moteurs de recherche classiques.

Dans des secteurs très pointus comme la médecine ou le droit, les LLM démontrent une grande efficacité pour la recherche d'informations au sein d'importants volumes de documents. Ils sont capables de trier, de résumer de longs textes et de répondre avec précision à des questions très techniques. En somme, ils sont essentiels pour l'automatisation de toute tâche axée sur l'analyse et la génération de texte.

L'objectif final de l'ouvrage est d'ailleurs de démystifier le fonctionnement interne de ces assistants complexes en programmant pas à pas un LLM capable de générer du texte et de répondre à différents types de requêtes.

<div align="center">

![Figure 1.2](img/figure_1.2.png)

</div>
*Figure 1.2 : Les interfaces LLM permettent une communication en langage naturel entre les utilisateurs et les systèmes d’IA. Cette capture d’écran montre ChatGPT écrivant un poème selon les instructions d’un utilisateur.*

### 1.3 Étapes de construction et d’utilisation des LLM

Construire son propre LLM depuis zéro permet de comprendre en profondeur son fonctionnement et ses limites, et d'acquérir les compétences nécessaires pour adapter des modèles open source à des besoins spécifiques (en utilisant généralement la bibliothèque PyTorch). Bien que des LLM généralistes comme ChatGPT soient très polyvalents, des modèles personnalisés et ciblés sur un domaine précis (comme la finance ou la médecine) peuvent souvent les surpasser.

Le développement de modèles sur mesure offre des avantages majeurs pour les entreprises : une protection totale de la confidentialité des données (évitant le partage avec des tiers), une réduction des coûts et de la latence (notamment grâce au déploiement local de petits modèles), ainsi qu'une autonomie complète sur le contrôle et les mises à jour.

La création d'un LLM repose sur un processus fondamental en deux étapes : le préentraînement (pretraining) suivi de l'affinage (fine-tuning).

La première étape, le **préentraînement**, s'effectue sur un immense corpus de textes « bruts » (non annotés). Cette phase utilise un apprentissage auto-supervisé où le modèle crée ses propres étiquettes en apprenant simplement à prédire le mot suivant. Le résultat est un modèle initial appelé « modèle de base » ou « foundation model » (comme GPT-3), doté d'une compréhension générale du langage, capable de compléter du texte et de réaliser de nouvelles tâches à partir de très peu d'exemples (*few-shot learning*).

<div align="center">

![Figure 1.3](img/figure_1.3.png)

</div>
*Figure 1.3 : Le préentraînement d’un LLM consiste à prédire le mot suivant sur de grands jeux de données textuels. Un LLM préentraîné peut ensuite être affiné à l’aide d’un plus petit jeu de données annoté.*

\
La seconde étape, l'**affinage (fine-tuning)**, sert à spécialiser ce modèle de base. Il s'agit d'un entraînement supplémentaire sur un jeu de données plus restreint mais, cette fois-ci, spécifiquement annoté. L'affinage se divise généralement en deux approches très populaires :
- Le *fine-tuning par instruction* : le modèle est entraîné sur des paires instruction-réponse (par exemple, une consigne de traduction et le texte traduit correspondant).
- Le *fine-tuning pour la classification* : le modèle s'entraîne à associer des textes à des étiquettes spécifiques (par exemple, trier des e-mails en "spam" ou "non-spam").

### 1.4 Introduction à l’architecture transformer

La majorité des LLM actuels reposent sur l'architecture **transformer**, apparue en 2017. Le transformer d'origine, créé pour la traduction automatique, se compose de deux modules principaux :
- Un **encodeur** qui analyse le texte d'entrée et le transforme en représentations numériques (vecteurs) intégrant le contexte.
- Un **décodeur** qui utilise ces vecteurs pour générer consécutivement le texte de sortie.

Le cœur de ce système est le mécanisme d'**auto-attention** (*self-attention*), qui permet au modèle d'évaluer et de pondérer l'importance de chaque mot d'une séquence par rapport aux autres, capturant ainsi efficacement le contexte global et les dépendances lointaines.

<div align="center">

![Figure 1.4](img/figure_1.4.png)

</div>
*Figure 1.4 : Représentation simplifiée de l’architecture transformer d’origine, un modèle de deep learning pour la traduction automatique.*

\
De l'architecture d'origine ont dérivé deux grandes familles de modèles, adaptées à des tâches distinctes :
- **BERT (Bidirectional Encoder Representations from Transformers)** : Ce modèle repose uniquement sur le sous-module **encodeur**. Il s'entraîne en prédisant des mots masqués, ce qui le rend particulièrement redoutable pour les tâches de classification de texte (analyse de sentiment, catégorisation de documents, détection de toxicité).
- **GPT (Generative Pretrained Transformers)** : Ce modèle se concentre sur le sous-module **décodeur**. Orienté vers la génération de texte (prédiction du mot suivant), il est très polyvalent. Il est notamment capable de résoudre des tâches pour lesquelles il n'a pas été explicitement entraîné avec peu ou pas d'exemples préalables (*few-shot* ou *zero-shot learning*).

<div align="center">

![Figure 1.5](img/figure_1.5.png)

</div>
*Figure 1.5 : Représentation visuelle des sous-modules encodeur et décodeur du transformer. À gauche, BERT (encodeur pour la classification). À droite, GPT (décodeur pour la génération).*

<div align="center">

![Figure 1.6](img/figure_1.6.png)

</div>
*Figure 1.6 : En plus de la complétion de texte, les LLM de type GPT peuvent résoudre diverses tâches en fonction de leurs entrées, sans nécessiter de réentraînement (few-shot ou zero-shot setting).*

\
*(Note : Dans cet ouvrage, le terme "LLM" désigne par convention les modèles basés sur les transformers, bien qu'historiquement d'autres architectures existent).*

### 1.5 Utilisation de grands jeux de données

Les capacités remarquables des LLM proviennent de l'utilisation de corpus textuels colossaux, comprenant des milliards de "tokens". Un token désigne une unité de texte analysée par le modèle (souvent un mot ou un signe de ponctuation). 

À titre d'exemple, le modèle de base GPT-3 a été entraîné sur environ 300 milliards de tokens, eux-mêmes extraits d'un jeu de données gigantesque filtré (CommonCrawl, livres, Wikipédia). C'est la taille et la diversité de ces données d'entraînement qui permettent au modèle d'acquérir une culture générale et une compréhension très fine de la sémantique et de la syntaxe.

Ces réseaux gigantesques sont appelés "modèles de base" (ou *foundation models*). Leur **préentraînement** initial est en revanche extrêmement coûteux en ressources informatiques (estimé à environ 4,6 millions de dollars pour GPT-3). 

Heureusement, de nombreux modèles de base sont proposés en open source. Il est donc possible de récupérer leurs paramètres (les "poids") préentraînés. L'essentiel à retenir ici est qu'un développeur peut ignorer l'étape onéreuse du préentraînement en réutilisant ces modèles open source, et se concentrer directement sur l'affinage (*fine-tuning*) pour ses propres tâches, des calculs qui restent à la portée d'un matériel informatique grand public.

### 1.6 Un regard plus précis sur l’architecture GPT

Les modèles GPT (Generative Pre-trained Transformer) ont été conçus par OpenAI. ChatGPT, par exemple, est issu d'un affinage d'une version de base très volumineuse (GPT-3) sur un jeu de données d'instructions. Malgré leur polyvalence (correction, classification, traduction), l'apprentissage fondamental de ces modèles repose sur une tâche étonnamment simple : la prédiction du mot suivant.

<div align="center">

![Figure 1.7](img/figure_1.7.png)

</div>
*Figure 1.7 : Dans la tâche de préentraînement de prédiction du mot suivant pour les modèles GPT, le système apprend à prédire le mot à venir dans une phrase en regardant les mots qui le précèdent.*

\
Cette prédiction du mot suivant est une forme d’apprentissage auto-supervisé. Le texte lui-même fournit la solution (l'étiquette), ce qui permet de s'affranchir de la très coûteuse annotation manuelle par des humains et d'utiliser d'immenses jeux de données textuelles non annotés. 

Contrairement au transformer d'origine (qui combinait encodeur et décodeur), l'architecture de GPT est plus simple : elle n'utilise que la partie **décodeur**. Le modèle génère le texte de façon itérative, un mot à la fois de gauche à droite, en réutilisant ses propres sorties précédentes comme entrées. C'est ce qu'on appelle un modèle **autorégressif**. Bien que simplifiée dans sa structure de base, cette architecture est poussée à une échelle gigantesque (GPT-3 possède 96 couches et 175 milliards de paramètres).

<div align="center">

![Figure 1.8](img/figure_1.8.png)

</div>
*Figure 1.8 : L’architecture GPT n’utilise que la partie décodeur du transformer d’origine. Elle est conçue pour un traitement unidirectionnel, de gauche à droite, ce qui la rend particulièrement adaptée à la génération de texte et à la prédiction du mot suivant, générant le texte de façon itérative, un mot à la fois.*

\
L'un des aspects les plus fascinants de ces modèles est l'apparition de **comportements émergents**. Bien qu'entraînés uniquement à prédire le mot suivant, l'exposition à des données si vastes et diverses permet aux modèles GPT de "comprendre" et de résoudre des tâches complexes (comme la traduction) pour lesquelles ils n'ont jamais été explicitement programmés.

### 1.7 Construire un grand modèle de langage

La construction d'un LLM depuis la base, qui est l'objectif global de ce livre, s'articulera autour de trois grandes étapes :

1. **Préparation et architecture (Étape 1)** : Apprendre les bases du prétraitement des données et coder le mécanisme d'attention, cœur du LLM.
2. **Préentraînement (Étape 2)** : Coder et entraîner un LLM de type GPT sur un petit jeu de données (à des fins éducatives, le préentraînement réel sur de gros volumes étant extrêmement coûteux). Cette étape couvrira aussi le chargement de poids de modèles open source existants.
3. **Affinage / Fine-tuning (Étape 3)** : Prendre le LLM préentraîné et l'adapter pour qu'il réponde à des instructions spécifiques (assistant personnel) ou classe des textes.

<div align="center">

![Figure 1.9](img/figure_1.9.png)

</div>
*Figure 1.9 : Les trois grandes étapes pour coder un LLM sont : implémenter l’architecture du LLM et le processus de préparation des données (étape 1), préentraîner un LLM pour créer un modèle de base (étape 2), et affiner ce modèle de base pour en faire un assistant personnel ou un classificateur de texte (étape 3).*

### Résumé du chapitre

En conclusion, les LLM marquent une évolution majeure du traitement du langage en s'appuyant sur le Deep Learning et l'architecture Transformer (spécifiquement le mécanisme d'attention). Ils reposent sur une conception en deux temps : un préentraînement massif et auto-supervisé (produisant un modèle de base doté de propriétés émergentes), suivi d'un affinage ciblé permettant de surpasser les modèles généralistes sur des tâches métiers précises. Les modèles de génération comme GPT (les assistants virtuels tels que ChatGPT) se contentent d'utiliser exclusivement le module décodeur pour une prédiction itérative et autoregressive des mots.

---

> **💡 Note de réflexion : Distinction entre Apprentissage "Zero-Shot" et Comportement Émergent**
> 
> Bien que ces deux concepts soient intimement liés dans la littérature des LLM, ils désignent des dimensions distinctes de l'intelligence artificielle :
> 
> * **Le Comportement Émergent (Le phénomène intrinsèque) :** Il s'agit d'une propriété fondamentale inhérente au modèle lui-même, découlant directement de son changement d'échelle (ce qu'on appelle les *scaling laws*). Lorsqu'un réseau de neurones atteint une taille critique (des milliards de paramètres) et qu'il est exposé à un volume colossal et hétérogène de données, il développe des facultés cognitives ou syntaxiques complexes (comme la traduction, la déduction logique ou la génération de code) pour lesquelles il n'a jamais été explicitement entraîné. L'objectif initial du modèle se limitait à la simple prédiction du mot suivant. Or, pour optimiser cette prédiction sur des textes complexes, le modèle est mathématiquement contraint d'assimiler la sémantique et la logique sous-jacente du langage. La compétence n'est donc pas programmée ; elle *émerge* d'elle-même de la profondeur du réseau et de la richesse des données.
> 
> * **L'Apprentissage Zero-Shot (Le paradigme d'inférence) :** Il désigne quant à lui le mode d'interaction ou le cadre d'utilisation du modèle. C'est la capacité pratique du LLM à exécuter une requête totalement inédite soumise par un utilisateur, sans que ce dernier n'ait besoin de fournir le moindre exemple (zéro exemple) illustrant la tâche à accomplir dans son "prompt".
> 
> **En synthèse :** Le *comportement émergent* explique la mécanique interne et structurelle du LLM (le "pourquoi" une machine conçue pour prédire des mots devient soudainement capable de traduire de l'anglais au français). L'*apprentissage zero-shot* est la manifestation opérationnelle de ce phénomène : c'est précisément grâce à l'émergence de ces compétences latentes que le modèle peut traiter avec succès une instruction "à froid" formulée par l'utilisateur.


