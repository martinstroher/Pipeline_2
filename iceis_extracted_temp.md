# **LLM-Driven Ontology Learning: From Term Extraction to Upper-Level** **Categorization for building Information System models**

Keywords: Ontology, Ontology Learning, Large Language Models, Natural Language Definitions


Abstract: Knowledge-intensive Information Systems support high-value corporate tasks. However, building knowledge
models requires manual selection and semantic clarification of terminology — a costly bottleneck in enterprise ontology engineering. This work explores automating conceptual modeling—terminology selection and
classification into upper-level meta-types using general-purpose Large Language Models (LLMs) to produce
precise semantic artifacts without specialized fine-tuning. We introduce an empirical framework comparing
four strategies: LLM + Corpus, LLM Generated, NER Model + Corpus, and TF-IDF, applied to 40 Brazilian
Pre-Salt scientific articles. Classification is standardized by leveraging the LLM to generate Natural Language
Definitions (NLDs) before categorization into a multi-layered hierarchy integrating Basic Formal Ontology
(BFO), GeoCore, and GeoReservoir. Resulting artifacts underwent blind validation by domain experts. Results demonstrate that non-specialized LLMs produce higher semantic quality and precision in term extraction
and taxonomic categorization than traditional statistical methods by leveraging NLDs. The LLM + Corpus
pipeline proved the most robust strategy. Ultimately, this work presents a validated paradigm for modernizing
ontology building through strategic, enterprise-scale automation during conceptual modeling.


## **1 INTRODUCTION**

The design of knowledge-intensive Information Systems depends critically on the quality of their underlying conceptual models. Ontologies, as formal and
explicit specifications of a shared conceptualization
(Studer et al., 1998), provide the structural backbone
for these systems and support various applications in
knowledge organization within the enterprise environment. The use of domain ontologies has demonstrated value in enhancing semantic clarity, thereby
eliminating ambiguities across data models, software
applications, and information retrieval systems. This
perceived value increases the number of ontology engineering projects in corporate environments, which
are built upon specialized terminology formalized in
ontology artifacts.
In this work, we focus on the construction of a domain ontology designed to serve as the core structural
backbone for an industrial software application that
retrieves and analyze analogous deposits in petroleum
pre-salt exploration in the industry — highly productive deep-water reservoirs that represent some of
the most significant energy discoveries of the last
decades. Based on our ontological framework, reservoir information from diverse heterogeneous sources
is stored in a uniform database for similarity analysis.
Despite recent advances, Ontology Engineering
still faces challenges in optimizing the ontology


building process. Figure 1, extracted from the widely
used NeOn Methodology for ontology engineering
(Su´arez-Figueroa et al., 2011), illustrates the entire
ontology construction scenario. The steps of terminology elicitation and clarification of the particular
semantics of the vocabulary remain bottlenecks in the
projects, since they require the involvement of highly
specialized human resources for extended labor sessions. This challenge is even more critical for enterprise applications. We aim to automate or reduce human intervention in the steps ”Non Ontological Resource Reuse”, ”Non Ontological Resource Reengineering”, and ”Conceptualization” as defined in the
NeOn methodology.
Non-ontological resource reuse begins with selecting terminology sources that determine the entities to be modeled. To ensure semantic interoperability and future expansion, these vocabularies are defined within a hierarchical structure (Guarino, 1998):
spanning from _top-level ontologies_ (foundational categories) through _core ontologies_, down to specialized
_domain ontologies_ . Therefore, we aim to build a robust domain ontology specialized from both a top and
a core ontology.
However, populating this robust framework traditionally relies on manual knowledge acquisition —
a laborious bottleneck that cannot keep pace with
the exponential growth of unstructured data. While
Large Language Models (LLMs) offer a potential so

Figure 1: NeOn scenarios for building ontologies. Adapted from (Su´arez-Figueroa et al., 2011), p. 13



lution to accelerate this process, approaches relying
on domain-specific fine-tuning (Sousa et al., 2023),
(Lo et al., 2024) often shift the resource burden, necessitating the curation of costly, manually annotated
datasets and demanding substantial computational infrastructure.

This work investigates a more accessible and generalizable path: applying a general-purpose foundation model without domain-specific fine-tuning. We
raise a critical research question: Can a non-specialist
LLM produce semantically precise artifacts, such as
term extraction and categorization into formal, multilayered ontological hierarchies, effectively reducing
human intervention? We design and compare four
distinct pipelines for term extraction (1) CorpusExtractive LLM, (2) LLM Generated, (3) NER Model
+ Corpus, and (4) a traditional TF-IDF baseline. We
then pivot on the LLM’s capacity to generate highquality Natural Language Definitions (NLDs) to improve the automated classification process (Lopes Junior, 2025) by using the LLM itself. The framework is
applied to the complex domain of Brazilian Pre-Salt
petroleum geology using a curated corpus. The results are then subjected to a blind empirical validation
by domain experts to assess quality, leading to a preliminary conceptual artifact based on terms deemed



relevant and correctly categorized.
The remainder of this work is organized as follows: Section 2 reviews foundational literature; Section 3 details our comparative empirical approach;
Section 4 describes the four pipelines, NLD generation, categorization into upper-level ontologies, and
blind validation steps; Section 5 presents the empirical findings and discussion; and Section 6 concludes
the study, summarizing insights and next steps.

## **2 BACKGROUND**


This section establishes the foundational knowledge
necessary for understanding the proposed approach
and reviews the current state-of-the-art literature rele
vant to LLM application in ontology engineering.


**2.1** **Ontology Learning**


Ontology Engineering’s primary bottleneck remains
knowledge acquisition — the manual, expert-driven
process of identifying relevant concepts, relations,
and axioms from domain-specific sources. This has
led to the sub-field of Ontology Learning, which seeks


to automate this process, primarily from unstructured
text (Khadir et al., 2021). This process can be modular, using specialized tools for individual components.
Our work focuses on the foundational upstream tasks
of term extraction and categorization, reviewing the
evolution of methods for these specific tasks.


**2.2** **Ontology Learning and Machine**
**Learning**


Traditional automated approaches to term extraction
have relied on statistical methods, such as the one
proposed by (Garcia et al., 2020b), which utilized
frequency-based analyzes (e.g., TF-IDF) to extract
and rank terms from a domain corpus. This work underscored the continued, critical role of domain expert validation. However, these methods fail semantically, unable to separate critical domain concepts
from noise

Embedding-based methods progressed from early
WordNet-based (Miller, 1994) approaches (Schmidt
et al., 2019), (Schmidt, 2020) that were limited by
general-purpose embeddings. The subsequent introduction of contextual embeddings (e.g., BERT (Devlin et al., 2019)) enhanced semantic capability for
downstream tasks like alignment and subsumption
prediction (Chen et al., 2023), (He et al., 2022), including methods utilizing top-level ontology context
(e.g.., the rdfs:comment field) (Sousa et al., 2023)
to improve automatic classification of domain concepts. However, these specialized models still require
costly domain-specific fine-tuning and manually labeled data, reintroducing a significant human bottleneck.


**2.3** **Ontology Learning and Large**
**Language Models**


Systematic reviews confirm the significant impact of
deep learning and LLMs across the ontology engineering lifecycle, from requirements specification to
maintenance (Amalki et al., 2025). This growing
trend underscores the potential of LLMs to reduce
the heavy reliance on human domain experts (Kommineni et al., 2024).
Different LLM model families were evaluated

for term typing, taxonomy discovery, and nontaxonomic relation extraction in zero-shot settings by
(Babaei Giglou et al., 2023), indicating that in their
pre-trained state, LLMs performed worst on term typing overall and struggled with ontology construction.
The LLMs4OL 2024 Challenge (Giglou et al., 2024)
established the first community-wide benchmarking



initiative for LLMs in ontology learning, further validating the field’s growing commitment to systematic
evaluation.

Fine-tuning approaches have emerged as a relevant strategy. (Lo et al., 2024) introduced an end-toend ontology learning method ( _OLLM_ ) that fine-tunes
LLMs with custom regularization to reduce overfitting on high-frequency concepts, avoiding the limitations of subtask-based composition. However, finetuning requires substantial labeled training data and
domain-specific annotation effort, which may not be
feasible in all ontology engineering contexts.
The capacity of LLMs to generate OWL ontology drafts directly from ontological requirements was
investigated by (Saeedizade and Blomqvist, 2024),
highlighting that advanced LLMs can produce OWL
suggestions comparable to those of novice human
modelers.. Aligning with these findings, (Lippolis
et al., 2025b) evaluated the generation of OWL ontology drafts from ontological requirements through
the usage of advanced prompting techniques, significantly outperforming novice human engineers in ontology modeling results, but still indicating the necessity for human oversight. _NeOn-GPT_ was introduced by (Fathallah et al., 2024): a semi-automatic
ontology learning pipeline that combines the structured _NeOn_ ((Su´arez-Figueroa et al., 2015)) methodology with the generative capabilities of LLMs. The
authors highlight key limitations in its ability to produce formally expressive ontologies that are validated against the gold standard ontology used in the
study. Three prompting strategies - direct, sequential,
and sentence-by-sentence - were explored by (Bakker
et al., 2024), observing that while LLMs effectively
identify classes and individuals, they often struggle to
generate consistent properties.
High performance (over 90% macro F1-score) and
robustness of language models in classifying domain
entities into top-level ontology concepts using Natural Language Definitions (NLDs), even across multilingual settings, was demonstrated by (Lopes Junior,
2025). This work highlights that NLDs are the most
effective textual representation for this classification
task. This critical insight provides a path forward: no
work has yet been designed or empirically compared
to leverage this NLD-centric insight for the engineering task of extracting domain terms from a corpus
and classifying them against a formal, multi-layered
upper-level ontology hierarchy.


## **3 A NEW APPROACH FOR TERM** **EXTRACTION AND** **CLASSIFICATION**

The path forward follows the insight from (Lopes Junior, 2025) that classification is highly effective if provided with a high-quality Natural Language Definition. We designed an empirical experiment comparing four distinct term extraction pipelines: (1) LLM +
Corpus (2) LLM Generated, (3) NER + Corpus, and
(4) TF-IDF as a baseline.
All four pipelines converge on a single, standardized classification module that (1) leverages an LLM
to generate a Natural Language Definition (NLD) for
each term, and then (2) uses that NLD to perform
a complex categorization against a formal, multilayered upper-level ontology structure. This structure is built upon three integrated ontologies: BFO
(Basic Formal Ontology) (Arp et al., 2015), which
serves as the foundational ontology; GeoCore (Garcia et al., 2020a), a core ontology in the domain that
formally utilizes BFO; and GeoReservoir (Cicconeto
et al., 2022), a parallel domain ontology built upon
GeoCore. GeoReservoir has a highly significant overlap with the domain for which we intend to create this
new ontology, meaning it will likely serve a complementary role. The BFO and GeoCore were a natural choice to expand the approach of (Lopes Junior,
2025) and ensure further integration with the existing
geology ontologies.
Then we execute a blind, multi-expert empirical validation. This assessment quantitatively and
qualitatively evaluates the final conceptual artifacts
(Terms, NLDs, and Categories) from all pipelines.

## **4 METHODOLOGY**


We have organized the methodology of this work into
four phases. The first phase is the preparation of the
curated corpus. The second phase consists of term
acquisition (performed using four distinct pipelines).
The third phase involves generating natural language
definitions (NLDs) and classifying the terms. Lastly,
the fourth phase consists of a comparative assessment
of the four pipelines across three critical dimensions:
term extraction coverage, NLD accuracy, and taxonomic categorization.


**4.1** **Phase 1: Corpus Preparation**


The foundation of our experiment is a highly specialized corpus consisting of 40 curated scientific articles



in the field of the Brazilian Pre-Salt. A senior domain
specialist rigorously selected these articles to ensure
that the corpus represents the maximum of optimized
and authoritative knowledge available. This focused
approach ensures the corpus is peer-reviewed, representative, and authoritative for the subsequent term
extraction and classification tasks, providing the necessary informational density while aligning with the
limited human resources.

The method automatically extracted the text from
all documents using a simple Python script leveraging the open-source library pdftotext. We decided
to retain the entire extracted text, including abstracts
and references, because these sections often contain
high-value terms and concepts definitions critical for
ontology construction.


**4.2** **Phase 2: Competing Term**
**Acquisition Pipelines**


All LLM-based steps utilized the general-purpose,
non-specialized foundation model Gemini 2.5 Pro
throughout the process. This choice was justified
by both its advanced reasoning capabilities, which
are crucial for categorization tasks, and by its high
availability, ease of use, and streamlined API access,
ensuring robust, reproducible, and operationally efficient execution. Unless otherwise specified (i.e.,
Pipeline 3), we have maintained a temperature of 0
for all LLM steps to maximize output determinism.


 - **Pipeline 1 (Corpus-Extractive LLM):** This
pipeline uses the LLM in a purely extractive
mode. We prompted the model to read the text
of each of the 40 articles and extract all relevant

geological terms. The method aggregated the extracted terms through stemming before proceeding to the subsequent classification workflow.


 - **Pipeline 2 (LLM Generated):** This pipeline operates in a zero-shot, corpus-agnostic manner. We
prompted the LLM to generate a list of the 500
most important terms for the domain (Pre-Salt
and analogs) based solely on its internal, pretrained knowledge. A low, non-zero temperature
(T=0.2) was chosen as a principled heuristic. This
option introduces minimal stochasticity, allowing the model to explore a slightly broader range
of high-probability terms and mitigate the risk
of repetitive, overly deterministic output, while
maintaining a strong focus on factual relevance.


 - **Pipeline 3 (Hybrid NER-LLM):** This pipeline
tests a hybrid, cross-lingual approach. The
specialized multilingual NER model XLMRoBERTa Large (Moreira et al., 2025) ) was al

lowed to extract terms and then fine-tuned on
the PetroGeoNER petroleum-specialized corpus.
Then, the method processed the terms and their
respective NER labels, which were extracted by
this model.


 - **Pipeline 4 (Statistical Baseline):** This pipeline
replicates the traditional methodology (Garcia
et al., 2020b) and serves as our non-AI baseline.
It uses the TF-IDF (Term Frequency-Inverse Document Frequency) weighting strategy, one of the
most popular and effective methods in information retrieval. Our method ran on the 40-article

corpus.


The raw outputs from all four pipelines showed
significant variability. To ensure systematic and reproducible aggregation, a normalization and consolidation step was implemented. This process involved
applying a stemmer for English to normalize terms to
their root form, followed by mapping each stem back
to its most frequent lexical form for human readability. The steps produced four distinct, consolidated,
and frequency-filtered lists, establishing the robust
empirical basis for the subsequent NLD-centric classification workflow.
The results from all distinct pipelines, along
with the source code and the specific LLM
prompts used for each step, are made available on
https://anonymous.4open.science/r/PreSaltOntology61FA/


**4.3** **Phase 3: NLD-Centric Classification**
**Workflow**


This phase implements a single, standardized workflow to process the candidate terms from all four
pipelines. We designed this workflow to explicitly test
the findings of (Lopes Junior, 2025) by using NLDs as
the pivot for classification.
All unique candidate terms from the four consolidated lists were fed into an NLD generation module,
leveraging the LLM to generate a canonical definition. We prompt the LLM with a system instruction
defining its role as a senior geoscientist and ontology
engineer. For each term, the LLM receives a prompt
template instructing it to generate a concise, technically clear, Aristotelian-style definition (e.g., ”X is a
Y that Z”) aligning with (Lopes Junior, 2025) findings.
The method converted the generated pairs (Term,
NLD) into JSON arrays and sent them to a dedicated
LLM classifier. Receiving one request per term, a
Chain-of-Thought (CoT) prompt instructed the model
to analyze the term and its NLD, and then classify it



into the most specific category possible. This classification strictly followed a fallback hierarchy: first,
GeoReservoir (more specialized ontology), then GeoCore (core ontology), and finally BFO (the top-level
ontology).


**4.4** **Phase 4: Specialist Evaluation**


To evaluate the performance of each pipeline, we designed a blind, multi-expert empirical validation performed by geologists specialized in the geological
Pre-Salt domain.


 - **Expected terms:** Before any sequential steps,
five domain specialists provided a reference set
of 100 terms (20 terms each) representing essential concepts for the Pre-Salt domain. This
group consisted of highly experienced professionals with advanced academic degrees (PhD
and Master’s) and extensive combined experience
spanning both petroleum industry operations and
petroleum academia.


 - **Sampling:** The method created a final sample
from each pipeline by selecting the 100 most relevant terms, along with their generated NLDs
and assigned categories, from each of the four
pipelines, based on its respective relevance ranking mechanism. These 400 entries were combined, anonymized, and randomized into a single
list. This fixed limit of 100 terms per pipeline
was a pragmatic decision driven by limited human resources and the recognized labor-intensive
nature of expert validation, ensuring the empirical
assessment remains feasible, rigorous, and nonexhaustive for the domain specialists.


 - **Quantitative Metrics:** The key metrics collected
for this study were:


**– Term Relevance:** To measure the quality of the
term acquisition (Relevant, Irrelevant, Invalid,
Unknown). The same five domain specialists
that provided the Expected terms participated
in this evaluation.

**– Definition Accuracy:** To measure the quality
of the NLD-generation step (Correct, Partially
Correct, Incorrect, Ambiguous / Too Vague,
Wrong Context, Unknown). Three from the
previous five domain experts participated in
this validation step: one highly specialized
researcher, with over 20 years of experience
in the Brazilian Pre-Salt domain; one PhD
specialist in Pre-Salt reservoir characterization
with 17 years of industry experience; and one
Master’s degree holder, also active in the PreSalt field.


Table 1: Term Extraction Coverage Analysis (Recall %))


**Pipeline** **Recall (%)**


**LLM + Corpus** 60
**LLM Generated** 40
**NER + Corpus** 32
**TF-IDF** 25


Table 2: Extracted Terms Relevance Distribution per Pipeline


**Pipeline** **Relevant (%)** **Irrelevant (%)** **Unknown (%)** **Invalid (%)**


**LLM + Corpus** 97,8% 2,0% 0,2% 0,0%
**LLM Generated** 96,2% 3,8% 0,0% 0,0%
**NER + Corpus** 86,6% 11,8% 0,8% 0,8%
**TF-IDF** 46,0% 46,4% 1,4% 6,2%



**– Category Accuracy:** To measure the quality
of the final classification, using a metric to capture the accuracy and specificity (e.g., Correct
and Specific, Correct but Unspecific, Incorrect,
Unknown). The same three specialists from the
previous step participated in this evaluation.


This mixed-methods assessment enables an em
pirical comparison of the results from each pipeline.

## **5 RESULTS AND DISCUSSION**


This section presents the comparative evaluation of
the four ontology learning pipelines: LLM + Corpus,
LLM Generated, NER + Corpus, and TF-IDF. Based
on the assessment by domain specialists, we analyze
performance across four critical dimensions: term extraction coverage, terminological relevance, NLD accuracy, and taxonomic categorization accuracy.


**5.1** **Term Extraction Coverage**


The terms extracted by the pipelines were compared
against the expert-defined reference terms (20 terms
provided by each of the five specialists) deemed essential for the domain coverage. The LLM + Corpus pipeline achieved the highest coverage, followed
by LLM-generated, while NER + Corpus and TF-IDF
demonstrated significantly lower recall (Table 1).
A clear distinction emerged in entity type performance. While concrete geological objects (e.g.
’Calcite’, ’Coquina’) were widely detected by LLM
and NER pipelines, a systematic gap existed in the
extraction of petrophysical properties and attributes
(e.g. ’grain roundness’). This absence was attributed
to strict prompt constraints prioritizing ”conceptual
classes” and excluding numerical values, confirming



model adherence rather than capability limitations.
Future iterations should explicitly request properties
to mitigate this.
A “granularity gap” was also noted, where automated extraction favored generic terms over the specific compound terms that experts had expected. Notably, TF-IDF exhibited the lowest efficacy, failing to
capture low-frequency terms that are highly relevant
(e.g., ‘Stevensite’). In contrast, semantic LLM methods successfully retrieved these terms.
In summary, the LLM + Corpus pipeline proved
the most robust for conceptual entities, offering
the best balance between precision and abstraction.
While NER + Corpus showed resilience in detecting named entities filtered out by LLM prompt constraints, it lacked semantic depth.


**5.2** **Term Relevance Evaluation**


Table 2 reveals a distinct dichotomy between semantic (LLM-based) and statistical/syntactic (TFIDF/NER + Corpus) extraction methods. The LLM +
Corpus pipeline achieved the highest overall and consensus relevance. This result contrasts sharply with
the TF-IDF baseline, which had introduced significant noise (46.4% Irrelevant, 6.2% Invalid).
Furthermore, the analysis of evaluator consensus
in Table 3 highlights a qualitative advantage of LLMs.
While the NER + Corpus pipeline achieved a respectable relevance rate, it generated a high volume of
’Mixed Feelings’ - this category signifies terms where
evaluators lacked consensus, being deemed simultaneously relevant and irrelevant by different specialists.

Finally, a subtle but important distinction emerged
between the LLM approaches. While both methods
achieved high relevance (¿96%), the LLM + Corpus
pipeline proved superior by using source text as a se

Table 3: Term Relevance Consensus Metrics


**Pipeline** **Strictly Relevant** **Strictly Irrelevant** **Mixed Feelings**


**LLM + Corpus** 91 0 9
**LLM Generated** 88 0 12
**NER + Corpus** 66 1 33
**TF-IDF** 25 22 53


Table 4: Natural Language Definition Evaluation Overall


**Accuracy Status** **LLM + Corpus** **LLM Generated** **NER + Corpus** **TF-IDF**


**Correct** 53,7% 66,0% 57,0% 34,7%
**Partially Correct** 34,0% 24,3% 22,3% 22,0%
**Incorrect** 9,7% 6,7% 15,3% 9,0%
**Ambiguous or Too Vague** 1,0% 2,3% 2,0% 14,7%
**Unknown** 1,0% 0,3% 2,0% 15,7%
**Wrong Context** 0,7% 0,3% 1,3% 4,0%



mantic anchor, resulting in slightly but significantly
higher consensus (91 for LLM + Corpus against 88
for LLM-generated). It contributed to mitigating the
”domain blindness” of the LLM-generated approach,
demonstrating that corpus grounding can help capture
the specialized vocabulary of the Pre-Salt domain.


**5.3** **Natural Language Definition**
**Evaluation**


Considering the identical, context-devoid definition
prompt across all pipelines, performance divergence
stemmed solely from the input term nature. The LLM
Generated pipeline achieved the highest accuracy 4.
This was attributted to the model ”internal coher
ence”, where the model prioritized canonical, theoretical concepts (e.g., ’Dolomite’) for which it possesses
robust internal definitions.
The LLM + Corpus pipeline had a high rate of
’Partially Correct’ definitions. This occurred because complex, empirically extracted terms lacked
the source context necessary for the general LLM to
generate precise definitions, leading to generic outputs insufficiently nuanced for experts. This exposes the critical methodological limitation of context
blindness during the definition phase. These findings
demonstrate that defining empirically extracted terms
is significantly more demanding than defining theoretical ones. Expert feedback also highlighted a scientific currency limitation: definitions sometimes reflected outdated paradigms (e.g., microbial origin of
Pre-Salt carbonates), due to the LLM’s training data’s
temporal constrains. Future iterations must incorporate Retrieval-Augmented Generation (RAG) to address both issues by providing necessary context and



an updated stream of high-quality domain literature to
prevent obsolete paradigm propagation.


**5.4** **Taxonomic Categorization**

**Evaluation**


Taxonomic categorization evaluation results are
shown in Table 5. It reveals a fundamental trade-off

between terminological stability and domain validity
driven by the origin of the input terms. The LLM
Generated pipeline achieved the highest Strict Consensus. The model prioritized canonical, theoretical
concepts (e.g., ’Dolomite’) from its pre-training that
elicit unanimous expert agreement. Conversely, the
LLM + Corpus pipeline demonstrated superior overall
accuracy, achieving the highest Majority Consensus
with the lowest rejection rate. While real-world terms
extracted from reports often exhibit complexity that
sparks taxonomic debate, they represent a more accurate reflection of the Pre-Salt domain than the theoretically ”clean” of the generative approach, which saw
a higher rejection rate.
The significant proportion of terms achieving Not
Strict Consensus over Strict Consensus (Correct and
Specific) in LLM + Corpus pipeline (Table 6) exposes
a critical methodological bottleneck: context blindness. This suggests the LLM’s full reasoning potential is currently constrained by the data architecture.
This may also indicate that implementing RAG in the
NLD generation phase can improve categorization,
and elevate majority consensus instances to strict consensus. Additionally, the limitations of non-semantic
methods were evident. NER + Corpus showed high
disagreement (45% Majority Correct vs. 23% Strict
Correct and Specific), and TF-IDF yielded the high

Table 5: Category Accuracy Distribution


**Accuracy Status** **LLM + Corpus** **LLM Generated** **NER + Corpus** **TF-IDF**


**Correct & Specific** 67,00% 64,67% 61,67% 40,67%
**Correct but Inespecific** 15,00% 11,00% 20,67% 10,00%
**Incorrect** 16,33% 22,67% 14,33% 25,33%
**Unknown / Cannot Judge** 1,67% 1,67% 3,33% 24,00%


Table 6: Category Accuracy Consensus Metrics


**Consensus Metric** **LLM + Corpus** **LLM Generated** **NER + Corpus** **TF-IDF**


**Strict Consensus:**
Correct and Specific 39 43 23 17
Incorrect 3 9 3 4
Correct but Inespecific 1 1 0 0
Unknown 0 0 0 0


**Not Strict Consensus:**
Correct and Specific 32 23 45 19
Incorrect 8 14 6 17
Correct but Inespecific 10 8 10 5
Unknown 0 0 0 16


**Ambiguity:**
No consensus 7 2 13 22



est rate of ”Unknown” results.

The high acceptance in the Table 7 rate further
confirms that using the corpus acts as a decisive ’Validity Filter’ for the extracted terms. When accounting for partial success, the LLM + Corpus pipeline
achieved a total acceptable categorization rate (Majority Correct and Majority Partially Correct) of 71
and 11, underscoring its efficacy as the primary
knowledge acquisition tool when compared to LLMgenerated categorized terms.


**5.5** **Resulting Conceptual Artifact**


The experiment also produced a tangible output: a
set of validated conceptual artifacts that represent a
preliminary domain ontology for Brazilian Pre-Salt
petroleum reservoirs, as shown in Figures 2 and 3.
This preliminary pre-salt ontology is available at
https://anonymous.4open.science/r/PreSaltOntology61FA/

The final consolidation of results yielded a core set
of domain concepts that achieved unanimous consensus among the three expert evaluators regarding both
Term Relevance (Relevant) and Category Accuracy
(Correct and Specific). These terms were formally instantiated into our ontological hierarchy.
Although the generated ontology is not axiomatized and does not encompass the entire set of required dependent entities, such as properties or rela


tionships, this resulting ontology remains significant.
It demonstrates that a semi-automated pipeline, despite its current limitations, when properly designed
with NLD generation and expert-in-the-loop validation, can bridge the gap between raw text and formal
ontological structures as a short step in ontology engineering. It moves beyond term extraction to produce
semantically grounded, hierarchically consistent artifacts, thereby easing the most labor-intensive initial
phases of domain ontology construction.

## **6 CONCLUSION**


This study evaluated a methodological framework for
partially automating Ontology Engineering knowledge acquisition in specific domains. Comparing four
pipelines, the LLM + Corpus approach proved most
robust for term extraction, outperforming TF-IDF and
NER significantly. Furthermore, the novel application of NLDs as a pivot for automated classification
proved effective, enabling the mapping of raw terms
into a formal, multi-layered hierarchy (BFO, GeoCore, GeoReservoir).
Beyond the methodological insights, this process
yielded a tangible, expert-validated conceptual artifact that addresses the specific needs of the Brazilian
Pre-Salt domain. This ontology serves as a structural
foundation for a enterprise software application in the


Table 7: Final Category Accuracy: Majority Consensus (%)


**Majority Metric** **LLM + Corpus** **LLM Generated** **NER + Corpus** **TF-IDF**


**Majority Correct** 71 66 68 36
**Majority Partially Correct** 11 9 10 5
**Majority Disputed / Rejected** 18 25 22 59


Figure 2: Resulting conceptual artifact, containing the validated terms classified into the three hierarchical levels: BFO,
GeoCore, and GeoReservoir. Classes in bold indicate the extracted terms. In the left tree, Unconformity is subsumed by GeoCore:GeologicalBoundary; Evaporite, Microbialite, Shale and Source Rock are subsumed by GeoReservoir:SedimentaryRock; Hydrocarbon is subsumed by GeoCore:EarthFluid; and Organic Matter is subsumed by GeoCore:AmountOfRock. In the right tree, Substrate is subsumed by GeoReservoir:BodyOfRock.



petroleum industry, designed to retrieve and analyze
analogous petroleum deposits, enabling the integration of heterogeneous reservoir data into a unified,
queryable format.
Empirical validation revealed a trade-off: while
LLM Generated pipeline provided better definitions
(”internal coherence”) but less relevant terms, LLM +
Corpus provided more relevant results but struggled
with ambiguous terms and outdated paradigms. Future work will integrate RAG into the NLD generation phase to overcome this context blindness. RAG
is expected to improve reasoning to resolve ambiguity and ensure scientific currency, paving the way for
a scalable, semi-automated process.



While this article presents an overview of the
results, we have also made available all supplementary material, including the code, curated
corpus, the full list of categorized terms, as
well as the corresponding expert evaluations in
https://anonymous.4open.science/r/PreSaltOntology61FA/.

## **REFERENCES**


Amalki, A., Tatane, K., and Bouzit, A. (2025). Deep
learning-driven ontology learning: A systematic map

Figure 3: Resulting conceptual artifact, containing validated terms classified into the three hierarchical levels: BFO, GeoCore,
and GeoReservoir. Classes in bold indicate the extracted terms. In the left tree, Alkalinity, Permeability, Reservoir Quality and
Porosity are subsumed by BFO:Quality; Onlap is subsumed by GeoCore:GeologicalContact. In the right tree, Compaction,
Diagenesis, Dissolution, Erosion, Evaporation, Silicification and Subsidence are subsumed by GeoCore:GeologicalProcess;
Hauterivian, Early Cretaceous, Cretaceous, Barremian, Aptian and Albian are subsumed by GeoCore:GeologicalTimeInterval.



ping study. _Engineering, Technology & Applied Sci-_
_ence Research_, 15(1):20085–20094.

Arp, R., Smith, B., and Spear, A. D. (2015). _Building on-_
_tologies with basic formal ontology_ . Mit Press.
Babaei Giglou, H., D’Souza, J., and Auer, S. (2023).
Llms4ol: Large language models for ontology learning. In _International Semantic Web Conference_, pages
408–427. Springer.

Bakker, R. M., Di Scala, D. L., and de Boer, M. H. (2024).
Ontology learning from text: an analysis on llm performance. In _Proceedings of the 3rd NLP4KGC Inter-_
_national Workshop on Natural Language Processing_
_for Knowledge Graph Creation, colocated with Se-_
_mantics_, pages 17–19.
Chen, J., He, Y., Geng, Y., Jim´enez-Ruiz, E., Dong, H.,
and Horrocks, I. (2023). Contextual semantic embeddings for ontology subsumption prediction. _World_
_Wide Web_, 26(5):2569–2591.

Cicconeto, F., Vieira, L. V., Abel, M., dos Santos Alvarenga, R., Carbonera, J. L., and Garcia, L. F. (2022).
Georeservoir: An ontology for deep-marine depositional system geometry description. _Computers &_
_Geosciences_, 159:105005.



Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K.
(2019). Bert: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings_
_of the 2019 conference of the North American chapter_
_of the association for computational linguistics: hu-_
_man language technologies, volume 1 (long and short_
_papers)_, pages 4171–4186.
Fathallah, N., Das, A., Giorgis, S. D., Poltronieri, A.,
Haase, P., and Kovriguina, L. (2024). Neon-gpt: a
large language model-powered pipeline for ontology
learning. In _European Semantic Web Conference_,
pages 36–50. Springer.

Garcia, L. F., Abel, M., Perrin, M., and dos Santos Alvarenga, R. (2020a). The geocore ontology: a core
ontology for general use in geology. _Computers &_
_Geosciences_, 135:104387.

Garcia, L. F., Rodrigues, F. H., J´unior, A. G. L., Kuchle,
R. d. S. A., Perrin, M., and Abel, M. (2020b). What
geologists talk about: Towards a frequency-based ontological analysis of petroleum domain terms. In _ON-_
_TOBRAS_, pages 190–203.
Giglou, H. B., D’Souza, J., and Auer, S. (2024). Llms4ol
2024 overview: The 1st large language models


for ontology learning challenge. _arXiv preprint_
_arXiv:2409.10146_ .

Guarino, N. (1998). _Formal ontology in information sys-_
_tems: Proceedings of the first international conference_
_(FOIS’98), June 6-8, Trento, Italy_, volume 46. IOS

press.
He, Y., Chen, J., Antonyrajah, D., and Horrocks, I. (2022).
Bertmap: a bert-based ontology alignment system. In
_Proceedings of the AAAI Conference on Artificial In-_
_telligence_, volume 36, pages 5684–5691.

Khadir, A. C., Aliane, H., and Guessoum, A. (2021). Ontology learning: Grand tour and challenges. _Computer_
_Science Review_, 39:100339.

Kommineni, V. K., K¨onig-Ries, B., and Samuel, S. (2024).
From human experts to machines: An llm supported
approach to ontology and knowledge graph construction. _arXiv preprint arXiv:2403.08345_ .
Lippolis, A. S., Saeedizade, M. J., Keskisarkka, R.,
Gangemi, A., Blomqvist, E., and Nuzzolese, A. G.
(2025a). Assessing the capability of large language models for domain-specific ontology generation. _arXiv preprint arXiv:2504.17402_ .
Lippolis, A. S., Saeedizade, M. J., Keskis¨arkk¨a, R., Zuppiroli, S., Ceriani, M., Gangemi, A., Blomqvist, E.,
and Nuzzolese, A. G. (2025b). Ontology generation using large language models. _arXiv preprint_
_arXiv:2503.05388_ .

Lo, A., Jiang, A. Q., Li, W., and Jamnik, M. (2024). Endto-end ontology learning with large language models.
_Advances in Neural Information Processing Systems_,
37:87184–87225.

Lopes Junior, A. G. (2025). _How to classify domain enti-_
_ties into top-level ontology concepts using language_
_models: a study across multiple labels, resources, do-_
_mains, and languages_ . PhD thesis, Universidade Federal do Rio Grande do Sul.

Miller, G. A. (1994). WordNet: A lexical database for English. In _Human Language Technology: Proceedings_
_of a Workshop held at Plainsboro, New Jersey, March_
_8-11, 1994_ .

Moreira, H., da Silva, P. F., Vieira, R., and Moreira, V.
(2025). Petrogeoner: A refined and unified dataset for
ner in the oil & gas domain. In _Simp´osio Brasileiro de_
_Tecnologia da Informac¸˜ao e da Linguagem Humana_
_(STIL)_, pages 259–271. SBC.
Saeedizade, M. J. and Blomqvist, E. (2024). Navigating
ontology development with large language models. In
_European Semantic Web Conference_, pages 143–161.
Springer.
Schmidt, D. (2020). Aligning top-level and domain ontologies.
Schmidt, D., Trojahn, C., and Vieira, R. (2019). Matching
bfo, dolce, gfo and sumo: an evaluation of oaei 2018
matching systems. In _ONTOBRAS_ .
Sousa, G., Lima, R., Vieira, R., and Trojahn, C. (2023).
Using bert models to automatically classify domain
concepts into dolce top-level concepts: A study of the
oaei ontologies. In _JOWO_ .
Studer, R., Benjamins, V., and Fensel, D. (1998). Knowl


edge engineering: Principles and methods. _Data &_
_Knowledge Engineering_, 25(1-2):161–197.
Su´arez-Figueroa, M. C., G´omez-P´erez, A., and Fern´andezL´opez, M. (2011). The neon methodology for ontology engineering. In _Ontology engineering in a net-_
_worked world_, pages 9–34. Springer.
Su´arez-Figueroa, M. C., G´omez-P´erez, A., and FernandezLopez, M. (2015). The neon methodology framework:
A scenario-based methodology for ontology development. _Applied ontology_, 10(2):107–145.


