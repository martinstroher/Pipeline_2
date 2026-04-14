import csv

# Taxonomy stats
tax = list(csv.DictReader(open('output/6_taxonomy.csv', encoding='utf-8-sig')))
intermediates = [r for r in tax if r['Is_Intermediate'] == 'True']
individuals = [r for r in tax if r['Relationship_Type'] == 'rdf:type']
classes = [r for r in tax if r['Relationship_Type'] == 'rdfs:subClassOf']
print(f'Taxonomy rows: {len(tax)}')
print(f'Intermediates: {len(intermediates)}')
print(f'Individuals: {len(individuals)}')
print(f'Classes (subClassOf): {len(classes)}')
print()
print('Sample intermediates:')
for r in intermediates[:20]:
    t, p, c = r['Term'], r['Parent_Term'], r['Category']
    print(f'  {t} -> {p} [{c}]')
print()
print('Sample individuals:')
for r in individuals[:20]:
    t, p, c = r['Term'], r['Parent_Term'], r['Category']
    print(f'  {t} -> {p} [{c}]')

# Frequency stats  
freq = list(csv.DictReader(open('output/3_filtered_top_terms.csv', encoding='utf-8-sig')))
freqs = sorted([int(r['Frequency']) for r in freq])
print(f'\nFrequency: min={freqs[0]}, max={freqs[-1]}, median={freqs[len(freqs)//2]}')
low = [r for r in freq if int(r['Frequency']) < 5]
print(f'Terms with freq < 5: {len(low)}')
print(f'Terms with freq < 10: {len([r for r in freq if int(r["Frequency"]) < 10])}')
print('Lowest 20:')
for r in sorted(freq, key=lambda x: int(x['Frequency']))[:20]:
    print(f'  {r["Readable_Term"]}: {r["Frequency"]}')

# Some potential "too generic" examples
print('\n=== Potentially generic terms ===')
cat = list(csv.DictReader(open('output/5_categorized_ontology.csv', encoding='utf-8-sig')))
generic_suspects = ['rock', 'mineral', 'process', 'material', 'fluid', 'water', 'time', 'object', 'property', 'structure', 'system', 'unit', 'body']
for r in cat:
    term = r['Term'].lower()
    if term in generic_suspects:
        print(f'  {r["Term"]} [{r["Category"]}]')
