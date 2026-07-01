# Comparatif RAGAS — baseline vs enrichi (SQL + agent)

- **Runs comparés** : `baseline_20260609_212658` (RAG texte seul, Excel aplati) vs `enriched_20260610_162008` (agent Pydantic AI : routing RAG / SQL).
- **Harness strictement identique** (`eval/evaluate_ragas.py`, seul le flag `--system` change) : 16 questions, juge `mistral-small-2506`, métriques `faithfulness` / `answer_relevancy` / `context_precision` / `context_recall`, 0 NaN sur les deux runs.
- `hors_couverture` **exclu des moyennes** (RAGAS n'évalue pas correctement les refus — voir leçons).
- **Aucun correctif appliqué après lecture des résultats** : les échecs ci-dessous sont documentés, pas patchés — corriger le système sur les questions mêmes du benchmark reviendrait à optimiser sur le jeu de test (n=16, pas de jeu de validation séparé).

---

## 1. Scores globaux (hors `hors_couverture`)

| métrique | baseline | enrichi | delta | lecture |
|---|---|---|---|---|
| faithfulness | 0.387 | **0.821** | **+0.434** | les réponses citent les sorties d'outils au lieu de broder |
| answer_relevancy | 0.821 | 0.784 | −0.037 | léger recul, causes identifiées (refus corrects punis + artefacts, voir Q11/Q4) |
| context_precision | 0.137 | **0.643** | **+0.506** | les contextes SQL sont pertinents là où l'Excel aplati ne l'était jamais |
| context_recall | 0.286 | **0.619** | **+0.333** | **métrique-titre** : la preuve attendue est désormais dans le contexte |

## 2. Scores par catégorie

| catégorie | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| simple_chiffre | 0.283 → **0.875** | 0.899 → 0.941 | 0.000 → **0.750** | 0.000 → **0.750** |
| complexe_chiffre | 0.322 → **1.000** | 0.665 → 0.671 | 0.000 → **0.750** | 0.000 → **0.417** |
| bruite | 0.243 → **0.833** | 0.859 → 0.867 | 0.000 → **0.500** | 0.000 → **0.500** |
| simple_texte | 0.628 → 0.583 | 0.882 → **0.699** | 0.479 → 0.500 | 1.000 → **0.750** |
| *hors_couverture (hors agrégat)* | 0.091 → 0.833 | 0.450 → 0.923 | 0.500 → 0.000 | 0.000 → 0.500 |

**Lecture.** Le delta est exactement là où le SQL devait le mettre : le `context_recall` chiffré passe de **0.0 partout** à 0.75 (simple) / 0.42 (complexe). En face, une **légère régression `simple_texte`** (recall 1.0 → 0.75, relevancy 0.88 → 0.70) — pas du bruit : trois causes identifiées (Q1 : reformulation, Q4 : retrieval + génération, artefacts d'adaptateur), détaillées ci-dessous. La branche RAG est inchangée, mais **ce qu'on lui donne à manger ne l'est plus** (reformulation par l'agent).

## 3. Routing de l'agent (run enrichi)

12 × `sql`, 4 × `rag` — **zéro erreur de routing sur les questions couvertes** (texte → RAG, chiffré → SQL). Les 2 `hors_couverture` (attendu `none`) sont parties en SQL : sonder la base pour constater l'absence serait un comportement acceptable — le problème n'est pas le choix de l'outil mais le **maquillage du résultat** (Q15, Q16).

---

## 4. Détail par question — scores baseline → enrichi

Format : `F` faithfulness · `AR` answer_relevancy · `CP` context_precision · `CR` context_recall.

| # | id | cat. | outil | F | AR | CP | CR | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | txt_impressive_duo | texte | rag | 0.80 → **0.00** | 0.90 → 0.86 | 0.00 → 0.00 | 1.00 → **0.00** | ❌ régression |
| 2 | txt_reggie_miller | texte | rag | 0.46 → **1.00** | 0.85 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | ✅ |
| 3 | txt_nba_marketing | texte | rag | 1.00 → 1.00 | 0.93 → 0.94 | 0.92 → 1.00 | 1.00 → 1.00 | ✅ |
| 4 | txt_homecourt_analogy | texte | rag | 0.25 → 0.33 | 0.85 → **0.00** | 0.00 → 0.00 | 1.00 → 1.00 | ❌ (déjà KO en baseline) |
| 5 | num_count_over_2000 | chiffre | sql | 0.33 → **1.00** | 0.94 → 0.98 | 0.00 → **1.00** | 0.00 → **1.00** | ✅ |
| 6 | num_best_3pct | chiffre | sql | 0.00 → 0.50 | 0.90 → 0.95 | 0.00 → 0.00 | 0.00 → 0.00 | ❌ |
| 7 | num_top_assists | chiffre | sql | 0.13 → **1.00** | 0.84 → 0.91 | 0.00 → **1.00** | 0.00 → **1.00** | ✅ |
| 8 | num_top_rebounds | chiffre | sql | 0.67 → **1.00** | 0.91 → 0.93 | 0.00 → **1.00** | 0.00 → **1.00** | ✅ |
| 9 | num_best_plusminus | complexe | sql | 0.25 → **1.00** | 0.90 → 0.90 | 0.00 → **1.00** | 0.00 → **1.00** | ✅ |
| 10 | num_compare_okc_min | complexe | sql | 0.32 → **1.00** | 0.79 → 0.84 | 0.00 → 0.00 | 0.00 → **0.67** | ✅ |
| 11 | num_team_total_rebounds | complexe | sql | 0.10 → 1.00 | 0.97 → **0.00** | 0.00 → 1.00 | 0.00 → 0.00 | ⚠️ refus propre |
| 12 | num_top10_points_sum | complexe | sql | 0.63 → 1.00 | 0.00 → 0.95 | 0.00 → 1.00 | 0.00 → **0.00** | ❌ cas d'école |
| 13 | noisy_3pct | bruite | sql | 0.09 → 0.67 | 0.87 → 0.88 | 0.00 → 0.00 | 0.00 → 0.00 | ❌ |
| 14 | noisy_count | bruite | sql | 0.40 → **1.00** | 0.85 → 0.85 | 0.00 → **1.00** | 0.00 → **1.00** | ✅ |
| 15 | oob_last5 | hors couv. | sql* | 0.18 → 0.67 | 0.00 → 0.95 | 1.00 → 0.00 | 0.00 → 1.00 | ❌ dangereux |
| 16 | oob_home_away | hors couv. | sql* | 0.00 → 1.00 | 0.90 → 0.90 | 0.00 → 0.00 | 0.00 → 0.00 | ❌ |

\* attendu : `none` (refus).

**Bilan absolu du système enrichi : 7 ✅ · 1 ⚠️ · 6 ❌ (+2 hors agrégat)** — mais le bilan *relatif* est le vrai résultat : la baseline n'avait **aucune** réponse chiffrée correcte (10 questions à recall 0.0), l'enrichi en réussit 7 sur 10 avec des chiffres exacts au point près.

---

## 5. Commentaires par question : le coupable et la raison

### Q1 — txt_impressive_duo ❌ *La régression la plus instructive*

- **Constat** : réponse **correcte** (Wagner + Banchero) mais faithfulness/recall 0.0 — la preuve n'est pas dans les contextes récupérés.
- **Coupable** : la **reformulation de l'agent**. Vérifié dans Logfire : l'agent a appelé le tool RAG avec `"duo Orlando Magic fans encensent"` — la question compressée **en mots-clés**. Aucune information perdue, mais `mistral-embed` n'encode pas une liste de mots-clés comme une question : la requête a atterri dans une autre région de l'espace vectoriel. Signature dans les scores FAISS : **75.8 / 74.2 / 74.1 / 74.0 / 73.7** — cinq chunks en mouchoir de poche, dont 4 du mauvais thread. Un retrieval sain a un top-1 qui se détache (cf. Q2).
- **Et la bonne réponse, alors ?** Issue des **poids du modèle** (le duo n'est nulle part dans les contextes) — violation de l'instruction « fonde ta réponse UNIQUEMENT sur les outils ». **Le faithfulness 0.0 est un vrai positif** : réponse juste *sans preuve* = même mécanisme que la confabulation de Q15, le dé est juste tombé du bon côté.
- **Leçon** : le retriever est inchangé, mais l'agent a introduit un **nouveau facteur de variance** : la requête qui entre dans FAISS. « La formulation pilote le retrieval » (leçon de la Phase 1) s'applique aussi à la reformulation machine.
- **💡 Solution** : transmettre la question utilisateur **verbatim** au tool RAG (1 ligne dans la docstring du tool) — *reco #1*.

### Q2 — txt_reggie_miller ✅

- F 0.46 → 1.00 : la baseline « animait le débat » (réponse longue, brodée) ; l'agent répond court et ancré.
- **Artefact** : un fragment `{"question": ...}` collé en fin de réponse — trace de l'adaptateur `MistralCompatChatModel` (contenu Mistral chunké aplati). **Cette réponse crashait avant l'adaptateur** ; elle passe désormais avec un défaut cosmétique. Crash → artefact : bon échange.

### Q3 — txt_nba_marketing ✅

- Stable au plafond (F 1.0, CR 1.0) sur les deux systèmes. C'est aussi **la question qui plantait l'agent** avant l'adaptateur OpenAI-compat — elle valide le correctif en conditions réelles.

### Q4 — txt_homecourt_analogy ❌ *(déjà KO en baseline — et le recall 1.0 est un faux positif du juge)*

- **Constat** : réponse **fausse** (« aucune équipe NHL… ») — et, inspection des chunks à l'appui (traces), la **vraie preuve n'est pas dans les contextes**. Le seul passage « Oilers » récupéré est un encart r/hockey disant qu'ils *ont* l'avantage de la glace en finale — pas qu'ils ont joué les trois tours précédents à l'extérieur (l'affirmation de la référence). Le chunk où le fan fait l'analogie n'a pas été remonté.
- **Coupables en cascade** :
  1. le **retrieval** — signature plate à nouveau (79.4 / 79.1 / 78.8 / 78.0 / 77.7, dont 2 chunks hors sujet) : le bon passage n'est pas remonté ;
  2. la **génération** — a répondu « jamais arrivé » en lisant le fil **NBA** voisin (« Never happened… ») : confusion de périmètre NBA/NHL ;
  3. le **juge** — **recall 1.0 à tort, sur les deux runs** : référence créditée sur la simple co-occurrence Oilers / finale / avantage, sans vérifier l'affirmation précise. Les métriques contexte de `mistral-small` sont laxistes face au paraphrasé proche.
- **Aggravant enrichi** : relevancy 0.00, probablement plombée par l'artefact `{"question": ...}` en fin de réponse (cf. Q2).
- **Leçon** : un score de métrique n'est pas une preuve — la localisation d'une panne exige la **lecture des données brutes** : ce recall 1.0 aurait fait conclure, à tort, que le retrieval était sain.
- **💡 Solution** : deux leviers côté retrieval — **améliorer l'ingestion** (l'OCR des PDF scannés produit des chunks très bruités qui dégradent les embeddings : nettoyage du texte OCR, re-chunking) et/ou **élargir la recherche** (k > 5, re-ranking) pour faire remonter le passage clé. Côté génération, un modèle plus attentif au cadre (NBA vs NHL) aiderait. Le recall 1.0 fantôme, lui, relève du juge (§ Limites) — aucune retouche du système ne le corrige.

### Q5 — num_count_over_2000 ✅

- `SELECT COUNT(*) … pts > 2000` → **4**, exact. Le quatuor parfait (1.0 / 0.98 / 1.0 / 1.0) — le cas nominal du text-to-SQL.

### Q6 — num_best_3pct ❌

- **Constat** : « minimum 200 tentatives » est dans la question, la règle est dans le prompt SQL, le pattern est dans le few-shot (`fta >= 300`)… et la requête générée n'a **pas de WHERE** → Alondes Williams, 100 % (volume infime). Référence : LaVine 44.6 %.
- **Coupable : la génération SQL** — **vérifié dans Logfire** : le tool a reçu `"Quel joueur a le meilleur pourcentage à 3 points (minimum 200 tentatives) cette saison régulière NBA ?"`. La reformulation est innocente (le filtre a survécu, la question a même été *enrichie* du cadre temporel). C'est le générateur SQL qui a ignoré le filtre malgré la règle explicite du prompt et le few-shot `fta >= 300` qui montre exactement ce pattern : l'exemple ne suffit pas à garantir le transfert.
- **Demi-point** : le modèle a flairé l'anomalie (« semble anormalement élevé ») — l'instinct était bon, la requête non.
- **💡 Solution** : **modèle plus capable pour la génération SQL** *(reco #2)* — le few-shot existait déjà et la consigne *explicite* a quand même été ignorée : c'est l'instruction-following du petit modèle qui plafonne, pas le prompt.

### Q7 / Q8 / Q9 — top passes, top rebonds, meilleur +/- ✅✅✅

- Trae Young 882 · Ivica Zubac 1008 · SGA +12.1 — **exacts au point près** face aux références calculées depuis l'Excel. Q8 est *l'échec emblématique de la baseline* (le RAG répondait à côté) : recall 0.0 → 1.0, c'est l'avant/après du brief en une ligne.

### Q10 — num_compare_okc_min ✅

- L'agent a **décomposé seul** la comparaison en **2 requêtes successives** (top scoreur OKC, puis top scoreur MIN) — comportement émergent du tool-calling, non programmé. Réponse exacte (SGA 2485 vs Edwards 2180).
- CP 0.0 / CR 0.67 malgré la réponse parfaite : le juge évalue chaque contexte isolément contre la référence *globale* — chacun des deux contextes ne prouve que la moitié. Limite de lecture des métriques contexte sur les réponses multi-requêtes.

### Q11 — num_team_total_rebounds ⚠️ *Le refus propre, puni par le juge*

- **Constat** : `WHERE t.name = 'Oklahoma City'` — le nom exact en base est `Oklahoma City Thunder` → somme sur ensemble vide → `NULL` → l'agent répond **« information non disponible »**.
- **Coupable** : la **génération SQL** (littéral approximatif), ironique car la question fournissait le **code** `OKC` — `team_code = 'OKC'` aurait suffi.
- **Le point clé** : le comportement de repli est **exactement celui demandé** par nos instructions (ne pas inventer) — et le juge le sanctionne : **relevancy 0.00**. Même anti-corrélation que sur `hors_couverture` : **RAGAS punit les refus corrects**. Sans cette ligne, le −0.04 global d'answer_relevancy disparaît quasiment.
- **💡 Solution** : **préférer `team_code`** quand un code est fourni, et mapper vers les valeurs exactes de `teams.name` sinon — l'utilisateur peut employer n'importe quel surnom (« OKC », « le Thunder », « Oklahoma »…), le prompt doit rendre ce mapping explicite. *(reco #3)*

### Q12 — num_top10_points_sum ❌ *Le cas d'école du run*

- **Le film** : 1ʳᵉ requête `SUM(pts) … ORDER BY pts LIMIT 10` → **rejetée par PostgreSQL** (`GroupingError` : on ne peut pas trier ce qui est déjà agrégé en une ligne). La boucle Pydantic AI renvoie l'erreur au modèle, qui **réessaie seul** : `SUM(p.pts) OVER ()` — window function **syntaxiquement valide**… qui somme les **569 joueurs** (280 015) au lieu des 10 meilleurs (réf. 19 931). Réponse finale auto-incohérente : « 280 015 points » suivi de la liste des 10 joueurs dont la somme fait visiblement ~20 000.
- **Coupables (en cascade)** : la **génération SQL** (le pattern « agrégat d'un top-N » exige une sous-requête, absent des few-shots — composition à deux étages, faiblesse typique des petits modèles), la **boucle de retry** qui a transformé l'échec **bruyant** (PostgreSQL strict) en erreur **silencieuse** (SQL valide mais faux), **et la synthèse de l'agent** qui a livré une réponse **auto-incohérente** — 280 015 annoncé juste au-dessus d'une liste de 10 joueurs qui somme visiblement ~20 000 — sans tiquer.
- **La leçon en or** : **faithfulness 1.0 sur une réponse fausse** — la réponse est parfaitement fidèle à un contexte qui est lui-même faux. *Faithfulness mesure l'ancrage, pas la vérité.* Seul `context_recall` (0.0, comparaison à la référence) attrape l'erreur. D'où le faisceau des 4 métriques.
- **💡 Solution** : triple levier *(reco #2)* — few-shot **agrégat-sur-sous-requête**, **meilleur modèle de génération SQL** (la composition est la marche des petits modèles), et **meilleur modèle d'agent** pour la synthèse : un sanity check minimal (la somme annoncée vs la liste affichée juste en dessous) suffisait à repérer la contradiction.

### Q13 — noisy_3pct ❌

- « kel joueur a le meilleur pourcentage a 3pts?? » → SQL syntaxiquement correct (robustesse au bruit OK) mais sans filtre de volume → Alondes Williams 100 %.
- **Coupable** : partagé. La question bruitée ne mentionne **pas** le filtre (contrairement à Q6) — la référence attend un filtre *implicite* (bon sens basket). La **génération SQL** n'a pas d'heuristique de volume minimal ; la **conception de la référence** suppose une inférence que le système ne fait pas. À discuter dans le RAPPORT comme limite de conception d'éval autant que du système.
- **💡 Solution** : **heuristique de volume minimal** sur les ratios (`*_pct`) dans le prompt SQL quand la question n'en précise pas (ex. tentatives ≥ seuil ligue). *(reco #5)*

### Q14 — noisy_count ✅

- « kombien de joueur on marké plus de 1.5k pts cet saison?? » → `COUNT(*) … pts > 1500` → **23**, exact. La robustesse orthographique est **gratuite** avec un LLM (vs un parseur classique) — bon argument d'architecture.

### Q15 — oob_last5 ❌ *Le mode d'échec le plus dangereux*

- **Constat** : la base n'a **aucun** détail par match. Le modèle a généré une requête *saison* (filtre `fg3a >= 200` correct !) et présenté le résultat comme « sur les **5 derniers matchs** » — donnée vraie, **étiquette fausse**, aplomb total.
- **Coupable : une responsabilité de cohérence non assignée — défaut de design, pas d'un seul maillon.** Le prompt SQL dit *« Génère UNE requête répondant à la question »* : le générateur n'a **aucun droit de refus**, il est structurellement obligé de produire quelque chose — ici il a silencieusement abandonné « 5 derniers matchs ». Puis la **synthèse** a rhabillé le résultat saison en « 5 derniers matchs » (**maquillage de cadre** : ni refus, ni chiffre inventé — une réinterprétation silencieuse). Dans l'architecture actuelle, **aucun étage n'est chargé de détecter qu'une question sort du schéma** — même mécanisme qu'en Q16.
- **RAGAS est aveugle** : relevancy 0.95, recall 1.0 (le juge « valide » contre une référence qui dit… que la donnée n'existe pas — les scores n'ont plus de sens sur les refus attendus). Validation magistrale de l'exclusion de `hors_couverture` de l'agrégat + jugement qualitatif.
- **💡 Solution** : garde-fou de périmètre dans les **instructions de l'agent** — règle de granularité (refuser toute question exigeant le détail de matchs précis : sous-période, date, domicile/extérieur, adversaire) avec l'exception dérivable (« par match » = total ÷ `gp` : répondable). *(reco #4 ; la température n'est pas le levier — l'erreur n'est pas un aléa d'échantillonnage.)*

### Q16 — oob_home_away ❌

- **Constat** : pour « domicile vs extérieur » (donnée inexistante), le modèle a **inventé une liste d'équipes "à domicile"** dans un `CASE WHEN ('BOS','GSW','LAL','LAC','PHX')` — hallucination structurelle dans le SQL lui-même.
- **Coupable** : le même **défaut de design qu'en Q15** (responsabilité hors-couverture non assignée) — le générateur SQL, sans droit de refus, va cette fois jusqu'à **fabriquer** une sémantique inexistante (la liste d'équipes « domicile » inventée). La synthèse a, elle, *signalé* l'incohérence (« valeurs inversées ou incomplètes… précise ta demande ») mais a quand même livré les chiffres. **Demi-point** pour l'instinct de doute.
- **💡 Solution** : même garde-fou qu'en Q15 — l'agent refuse **avant** d'appeler le tool : le générateur n'est plus jamais exposé au hors-schéma, la fabrication disparaît. *(reco #4 ; accessoirement #2.)*

---

## 6. Leçons transverses

1. **Faithfulness ≠ justesse** (Q12) : la métrique mesure la fidélité de la réponse *au contexte*, pas la vérité du contexte. Une chaîne peut être « fidèle à un contexte faux ». Le faisceau des 4 métriques (+ les références vérifiées) est indispensable.
2. **RAGAS punit les refus corrects** (Q11, Q15, Q16 et déjà la baseline) : relevancy 0 quand le système dit honnêtement « indisponible », scores flatteurs quand il confabule avec aplomb. → `hors_couverture` hors agrégat + jugement qualitatif : décision de Phase 2 doublement validée.
3. **La reformulation de l'agent est un facteur de variance du retrieval** (Q1 — innocentée en Q6, vérifié dans les traces : le filtre avait survécu à la reformulation) : le retriever est figé, mais la requête qui y entre ne l'est plus. Tout maillon LLM intercalé déplace le comportement des maillons suivants.
4. **Échec bruyant > échec silencieux** (Q12, et l'inverse en Q11) : PostgreSQL strict, la validation Pydantic, le rôle SELECT-only — tout ce qui refuse fort vaut mieux que tout ce qui devine. Le retry automatique peut *retransformer* un échec bruyant en erreur silencieuse : à surveiller.
5. **Le text-to-SQL échoue par composition, pas par syntaxe** (Q12, Q16) : les requêtes plates sont fiables ; les compositions (agrégat-sur-top-N) et les demandes hors schéma produisent du SQL plausible et faux. La marche est sémantique, pas syntaxique.
6. **L'observabilité a rendu chaque diagnostic possible** : reformulations (args des tool calls), requêtes SQL générées, scores FAISS — chaque « coupable » ci-dessus a été identifié dans une trace Logfire, span par span.

## 7. Recommandations actionnables (à valider sur de NOUVELLES questions, jamais sur ce benchmark)

| # | Recommandation | Cible | Coût |
|---|---|---|---|
| 1 | Passer la question utilisateur **verbatim** au tool RAG (docstring du tool : « pass the user's question unchanged ») | Q1 | 1 ligne |
| 2 | **Montée en gamme des modèles** : few-shot agrégat-sur-sous-requête + **modèle plus capable pour la génération SQL** (Q6/Q12/Q16 : instruction-following et composition) **et pour l'agent** (Q12 : une synthèse capable de repérer une réponse auto-incohérente) | Q6, Q12, Q16 | 3 lignes / config |
| 3 | Prompt SQL : **préférer `team_code`** quand un code est fourni ; sinon mapper vers les valeurs exactes de `teams.name` (l'utilisateur peut employer n'importe quel surnom) | Q11 | 2 lignes |
| 4 | Hors-couverture : **garde-fou porté par l'agent seul** — formulation « règle de granularité + exception dérivable » (pas de liste de mots-clés) ; générateur SQL inchangé ; **validation sur de nouvelles questions des deux côtés de la frontière** (hors-couverture vraies ET dérivables pièges, ex. « points par match ») ; risque résiduel documenté | Q15, Q16 | ~5 lignes |
| 5 | Heuristique de volume minimal sur les ratios (`*_pct`) quand la question n'en précise pas | Q13 | prompt SQL |

> La limite du juge (refus punis, recall laxiste — Q4, Q11, Q15) reste documentée en § Limites : c'est une
> limite d'**évaluation**, pas une action sur le système.

## 8. Limites du comparatif

- **Juge `mistral-small-2506`** (contrainte free tier) : scores indicatifs ; le **delta** est le signal, pas les valeurs absolues. Exemple concret : le **recall 1.0 de Q4 est un faux positif** (référence créditée sur une co-occurrence Oilers/finale/avantage alors que l'affirmation précise est absente des chunks) — détecté uniquement par lecture des contextes bruts.
- **Run unique, routing non déterministe** : une question borderline peut changer d'outil entre runs (temp 0.1 ≠ 0) ; le routing observé (12 sql / 4 rag) est cohérent mais pas garanti reproductible à l'identique.
- **n = 16** : les moyennes par catégorie reposent sur 2 à 4 questions — lire les lignes, pas seulement les agrégats.
- Artefact `{"question": ...}` dans 2 réponses texte (adaptateur OpenAI-compat) : pénalise probablement la relevancy de Q4 ; cosmétique mais mesurable.
