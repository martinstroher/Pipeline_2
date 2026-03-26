Synthesis: Highest-Value Ideas by Effort
Must-Do (< 30 min each, significant methodological impact)
#	Idea	Effort	Why it matters
1	Add few-shot examples to NLD + categorizer prompts (from test_dataset.json)	30 min each	Single highest-impact prompt change; 15-25% consistency improvement per literature
2	Fix retrieval query from "What is the definition of {term}?" to just "{term}"	5 min	Current question format adds BM25 noise and biases dense retrieval toward definitional passages only
3	Use "No additional context available." for Condition B instead of empty string	5 min	Empty string cues the model that context is "missing," biasing behavior
4	Use "No definition available." for Condition C instead of rewriting the prompt	5 min	Keeps prompt template identical, only NLD content changes — cleaner ablation
5	Bonferroni correction (3 pairwise → α=0.0167)	1 line	Required for 3-way comparison; committee will demand it
6	Report effect sizes for every test (Cohen's g, rank-biserial r)	5 min/test	With 769 terms, everything will be "significant" — effect sizes tell the real story
7	Add "output in English" to NLD prompt	1 min	Portuguese corpus may cause code-switching without it
8	Fix NLD system instruction — pass as system_instruction= param, not string concat	5 min	Categorizer does this correctly (line 104); NLD generator doesn't
Should-Do (1-2h each, strong thesis additions)
#	Idea	Effort	Why it matters
9	Subgroup analysis by Context_Used flag	0h (analysis only)	"Terms where RAG context was used had X% higher accuracy" — proves the mechanism, not just the outcome
10	Add Condition D (RAG context → categorizer, NO NLD intermediary)	1-2h	Eliminates strongest counterargument: "Why not skip NLDs and pass raw context?"
11	"Where RAG Helped/Hurt" qualitative analysis	2h	Most memorable thesis result — specific examples > aggregate p-values
12	Hallucination catalog (10-15 terms where Condition B got it wrong)	1h	Devastating evidence; more memorable than any statistical test
13	Automatic genus extraction from NLDs via regex	1.5h	Free proto-taxonomy from "X is a Y that Z" — addresses "flat vocabulary" criticism
14	Parametric recall experiment (give LLM a Pre-Salt description, measure how many of 769 terms it generates)	1.5h	Directly quantifies NeOn-GPT's weakness on YOUR domain
15	Category migration analysis (which terms change category across conditions)	30 min	Shows WHERE RAG helps — likely GeoReservoir-specific terms
Nice-to-Have (for polish)
#	Idea	Effort	Why
16	Confusion matrix heatmaps per condition	1h	High visual impact thesis figure
17	Increase BATCH_SIZE from 1 to 15 for categorizer	5 min	15x fewer API calls
18	Add confidence field (high/medium/low) to categorizer output	15 min	Free analysis signal
19	Log reranker scores per term for corpus coverage analysis	20 min	Correlate retrieval quality → NLD quality → categorization accuracy
20	Zenodo deposit for DOI/citability	1h (at the end)	Permanent citable artifact
Critical Warning from All Three Agents
The expert ground truth CSV does not exist in the repository. The ~1200-row expert evaluation data is referenced but not present. The 22-item test_dataset.json golden answers were LLM-synthesized. Layer 1 automated evaluation depends entirely on having real expert category labels. This must be added before the ablation runs.

On Condition D specifically
Agent 1 made a strong case for it: without Condition D, a reviewer can ask "Why not skip NLDs and pass raw RAG context directly to the categorizer?" Your answer would be handwaving. With Condition D (A > D), you prove NLDs are a better information representation than raw chunks. The four comparisons become:

A vs B → RAG improves NLDs
A vs C → NLDs improve categorization
B vs C → Even parametric NLDs help
A vs D → NLDs compress RAG context better than raw context
Cost: ~1h more API time for categorization only (no NLD generation for D)