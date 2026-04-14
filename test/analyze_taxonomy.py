"""Temporary taxonomy analysis script."""
import pandas as pd

tax = pd.read_csv('output/6_taxonomy.csv', encoding='utf-8-sig')

print('=== BASIC STATS ===')
print(f'Total rows: {len(tax)}')
print(f'Unique terms: {tax["Term"].nunique()}')
print(f'Intermediate nodes: {tax["Is_Intermediate"].sum()}')
print(f'rdf:type (individuals): {(tax["Relationship_Type"]=="rdf:type").sum()}')
print(f'rdfs:subClassOf: {(tax["Relationship_Type"]=="rdfs:subClassOf").sum()}')

print('\n=== CATEGORY DISTRIBUTION ===')
print(tax['Category'].value_counts().to_string())

print('\n=== INTERMEDIATE NODES BY CATEGORY ===')
inter = tax[tax['Is_Intermediate']==True]
print(inter['Category'].value_counts().to_string())

print('\n=== DEPTH ANALYSIS ===')
parent_map = dict(zip(tax['Term'], tax['Parent_Term']))

def get_depth(term, visited=None):
    if visited is None:
        visited = set()
    if term in visited:
        return 0
    visited.add(term)
    parent = parent_map.get(term)
    if pd.isna(parent) or parent == term or parent not in parent_map:
        return 0
    return 1 + get_depth(parent, visited)

depths = {t: get_depth(t) for t in tax['Term'].unique()}
depth_series = pd.Series(depths)
print(f'Max depth: {depth_series.max()}')
print(f'Mean depth: {depth_series.mean():.2f}')
print(f'Depth distribution:')
print(depth_series.value_counts().sort_index().to_string())

print('\n=== DEEPEST CHAINS (depth>=4) ===')
for term, d in sorted(depths.items(), key=lambda x: -x[1])[:10]:
    chain = [term]
    current = term
    for _ in range(10):
        p = parent_map.get(current)
        if pd.isna(p) or p == current or p not in parent_map:
            if not pd.isna(p):
                chain.append(p)
            break
        chain.append(p)
        current = p
    print(f'  depth={d}: {" -> ".join(chain)}')

print('\n=== NEAR-SYNONYM CANDIDATES ===')
from collections import defaultdict
# Group terms that share prefix/suffix
terms = sorted(tax['Term'].unique())
# Check for terms with shared stems
seen_groups = []
for i, t1 in enumerate(terms):
    words1 = set(t1.lower().split())
    for t2 in terms[i+1:]:
        words2 = set(t2.lower().split())
        overlap = words1 & words2
        if len(overlap) >= 2 and len(words1) >= 2 and len(words2) >= 2:
            ratio = len(overlap) / max(len(words1), len(words2))
            if ratio >= 0.6:
                p1 = parent_map.get(t1, '?')
                p2 = parent_map.get(t2, '?')
                cat1 = tax[tax['Term']==t1]['Category'].iloc[0] if len(tax[tax['Term']==t1]) > 0 else '?'
                cat2 = tax[tax['Term']==t2]['Category'].iloc[0] if len(tax[tax['Term']==t2]) > 0 else '?'
                print(f'  [{ratio:.0%}] "{t1}" ({cat1}) vs "{t2}" ({cat2})')

print('\n=== POTENTIAL CLASS/INDIVIDUAL ISSUES ===')
# Things that are rdf:type but look generic
individuals = tax[tax['Relationship_Type']=='rdf:type']
for _, row in individuals.iterrows():
    term = row['Term']
    # Check if term looks generic (no proper noun indicators)
    words = term.split()
    if not any(w[0].isupper() for w in words if len(w) > 2):
        print(f'  Generic-looking individual: "{term}" (cat: {row["Category"]})')

print('\n=== CHILDREN COUNT PER INTERMEDIATE ===')
child_counts = tax.groupby('Parent_Term').size().sort_values(ascending=False)
inter_terms = set(inter['Term'])
print('Top intermediates by child count:')
for parent, count in child_counts.head(30).items():
    is_inter = '(INTER)' if parent in inter_terms else ''
    print(f'  {parent}: {count} children {is_inter}')

print('\n=== CATEGORY OVERLAP CHECK ===')
# Terms categorised differently in taxonomy vs their parent's category
for _, row in tax.iterrows():
    parent = row['Parent_Term']
    if pd.notna(parent):
        parent_rows = tax[tax['Term']==parent]
        if len(parent_rows) > 0:
            parent_cat = parent_rows.iloc[0]['Category']
            if parent_cat != row['Category']:
                print(f'  Cross-category: "{row["Term"]}" ({row["Category"]}) -> parent "{parent}" ({parent_cat})')
