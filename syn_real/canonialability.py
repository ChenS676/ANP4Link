import matplotlib.pyplot as plt

# Data from the table
k = [4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 'All']
uncanonizable_eigenvectors = [1.21, 2.13, 3.48, 5.61, 8.39, 10.40, 11.68, 13.13, 13.75, 13.07, 12.63]
uncanonizable_graphs = [4.36, 9.68, 18.44, 33.02, 52.14, 65.37, 73.66, 84.93, 86.37, 86.48, 86.48]


# Replace 'All' with numeric value for plotting continuity
k_numeric = list(range(1, len(k)+1))

plt.figure(figsize=(7, 4))
plt.plot(k_numeric, uncanonizable_eigenvectors, marker='o', label='Uncanonizable Eigenvectors (%)')
plt.plot(k_numeric, uncanonizable_graphs, marker='s', label='Uncanonizable Graphs (%)')

# Replace x-ticks with actual labels (including 'All')
plt.xticks(k_numeric, k)
plt.xlabel(r'$k$')
plt.ylabel('Percentage (%)')
plt.title('Canonizability Results on ZINC Across $k$')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('cano.pdf')
