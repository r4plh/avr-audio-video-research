import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any
import plotly


class Visualizer:
    """Create interactive visualizations with Plotly"""

    def __init__(self):
        # Color palette for words
        self.colors = [
            '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
            '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
            '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
            '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
            '#ffffff', '#000000', '#ff6347', '#4682b4', '#32cd32',
            '#ff1493', '#00ced1', '#ff8c00', '#9370db', '#00fa9a',
            '#dc143c', '#00bfff', '#ffd700', '#ff69b4', '#87ceeb'
        ]

    def get_color_map(self, labels: List[str]) -> Dict[str, str]:
        """Create color mapping for unique labels"""
        unique_labels = sorted(list(set(labels)))
        color_map = {}
        for i, label in enumerate(unique_labels):
            color_map[label] = self.colors[i % len(self.colors)]
        return color_map

    def reduce_dimensions(self, embeddings: np.ndarray, method: str, n_components: int) -> np.ndarray:
        """
        Reduce embedding dimensions
        Args:
            embeddings: (n_samples, n_features) array
            method: 'pca' or 'tsne'
            n_components: 2 or 3
        """
        n_samples, n_features = embeddings.shape

        # Validate minimum requirements
        min_samples_required = max(n_components, 2)  # At least 2 samples for any reduction
        if n_samples < min_samples_required:
            raise ValueError(
                f"Not enough samples for {n_components}D {method.upper()}. "
                f"Got {n_samples} samples, need at least {min_samples_required}. "
                f"Please select more words or increase samples per word."
            )

        # Adjust n_components if needed
        max_components = min(n_samples, n_features)
        actual_components = min(n_components, max_components)

        if actual_components < n_components:
            print(f"Warning: Reducing from {n_components}D to {actual_components}D due to data constraints")

        if method == 'pca':
            reducer = PCA(n_components=actual_components, random_state=42)
            reduced = reducer.fit_transform(embeddings)
            variance = reducer.explained_variance_ratio_.sum()
            print(f"PCA variance explained: {variance:.2%}")

            # Pad with zeros if we had to reduce components
            if actual_components < n_components:
                padding = np.zeros((n_samples, n_components - actual_components))
                reduced = np.hstack([reduced, padding])

            return reduced, variance

        elif method == 'tsne':
            # t-SNE also needs enough samples
            if n_samples < 3:
                raise ValueError(
                    f"t-SNE requires at least 3 samples, got {n_samples}. "
                    f"Please select more words or increase samples per word."
                )

            # Perplexity should be between 5 and 50, and less than n_samples
            # Use heuristic: sqrt(n_samples) bounded by [5, 30]
            perplexity = max(5, min(30, int(np.sqrt(n_samples))))
            perplexity = min(perplexity, n_samples - 1)

            reducer = TSNE(
                n_components=actual_components,
                random_state=42,
                metric='cosine',
                max_iter=10000,
                n_iter_without_progress=1000,
                perplexity=perplexity
            )
            reduced = reducer.fit_transform(embeddings)
            print(f"t-SNE perplexity used: {perplexity}")

            # Pad with zeros if we had to reduce components
            if actual_components < n_components:
                padding = np.zeros((n_samples, n_components - actual_components))
                reduced = np.hstack([reduced, padding])

            return reduced, None

        else:
            raise ValueError(f"Unknown reduction method: {method}")

    def create_scatter_trace(self, data: np.ndarray, labels: List[str],
                            color_map: Dict[str, str], word: str,
                            show_legend: bool = True, dimensions: str = '3d') -> go.Scatter3d:
        """Create a single scatter trace for a word"""
        mask = [l == word for l in labels]

        if dimensions == '3d':
            return go.Scatter3d(
                x=data[mask, 0].tolist(),  # Convert to list to avoid binary encoding
                y=data[mask, 1].tolist(),
                z=data[mask, 2].tolist(),
                mode='markers',
                name=word,
                marker=dict(
                    size=5,
                    color=color_map[word],
                    opacity=0.7,
                    line=dict(width=0)
                ),
                showlegend=show_legend,
                legendgroup=word
            )
        else:  # 2d
            return go.Scatter(
                x=data[mask, 0].tolist(),  # Convert to list to avoid binary encoding
                y=data[mask, 1].tolist(),
                mode='markers',
                name=word,
                marker=dict(
                    size=8,
                    color=color_map[word],
                    opacity=0.7,
                    line=dict(width=0)
                ),
                showlegend=show_legend,
                legendgroup=word
            )

    def create_plots(self, results: Dict[str, Any], viz_type: str, dimensions: str) -> Dict[str, str]:
        """
        Create all requested plots
        Args:
            results: Dictionary with embeddings, labels, config
            viz_type: 'pca', 'tsne', or 'both'
            dimensions: '2d' or '3d'

        Returns:
            Dictionary of plot HTML strings
        """
        embeddings_dict = results['embeddings']
        labels = results['labels']
        layer_mode = results.get('layer_mode', 'individual')
        dac_metadata = results.get('dac_metadata', {})

        # Store variance info for PCA plots
        variance_info = {}

        # Validate we have data
        if not labels:
            raise ValueError("No samples found. Please check your configuration and try again.")

        if not embeddings_dict:
            raise ValueError("No embeddings extracted. Please check model and layer selection.")

        # Check that we have valid embeddings
        valid_embeddings = False
        for model_name, emb_types in embeddings_dict.items():
            if layer_mode == 'concatenate':
                if 'concatenated' in emb_types:
                    valid_embeddings = True
                    break
            else:
                for emb_type, emb_data in emb_types.items():
                    if len(emb_data) > 0:
                        valid_embeddings = True
                        break
            if valid_embeddings:
                break

        if not valid_embeddings:
            raise ValueError("No valid embeddings found. All embedding arrays are empty.")

        color_map = self.get_color_map(labels)
        unique_words = sorted(list(set(labels)))

        plots = {}

        # Determine which reduction methods to use
        methods = []
        if viz_type in ['pca', 'both']:
            methods.append('pca')
        if viz_type in ['tsne', 'both']:
            methods.append('tsne')

        # Determine which dimensions to use
        dimension_configs = []
        if dimensions == '2d':
            dimension_configs = [('2d', 2)]
        elif dimensions == '3d':
            dimension_configs = [('3d', 3)]
        else:  # 'both' or any other value
            dimension_configs = [('2d', 2), ('3d', 3)]

        for method in methods:
            for dim_name, n_components in dimension_configs:
                # Count total number of embedding types across all models
                if layer_mode == 'concatenate':
                    total_plots = len(embeddings_dict)  # One plot per model
                else:
                    total_plots = sum(
                        len(emb_types) for emb_types in embeddings_dict.values()
                    )

                # Calculate subplot grid
                n_cols = min(3, total_plots)
                n_rows = (total_plots + n_cols - 1) // n_cols

                # Create subplot specification
                if dim_name == '3d':
                    specs = [[{'type': 'scatter3d'} for _ in range(n_cols)] for _ in range(n_rows)]
                else:
                    specs = [[{'type': 'scatter'} for _ in range(n_cols)] for _ in range(n_rows)]

                # Create subplots - titles will be updated after variance calculation for PCA
                subplot_titles = []
                plot_index = 0

                for model_name, emb_types in embeddings_dict.items():
                    if layer_mode == 'concatenate':
                        # In concatenate mode, we have metadata about the concatenation
                        model_short = model_name.split('/')[-1]
                        if 'total_dim' in emb_types:
                            title = f"{model_short}<br>Concatenated ({emb_types['total_dim']}D)"
                        else:
                            title = f"{model_short}<br>Concatenated"
                        subplot_titles.append(title)
                    else:
                        for emb_type in sorted(emb_types.keys()):
                            # Create title
                            model_short = model_name.split('/')[-1]

                            # Add DAC metadata to title if available
                            if model_name in dac_metadata:
                                meta = dac_metadata[model_name]
                                n_cb = meta['n_codebooks']
                                emb_dim = emb_types[emb_type][0].shape[0] if len(emb_types[emb_type]) > 0 else '?'
                                title = f"{model_short}<br>{emb_type} ({emb_dim}D)<br>Codebooks: {n_cb}"
                            else:
                                title = f"{model_short}<br>{emb_type}"
                            subplot_titles.append(title)

                print(f"\n{method.upper()} - {dim_name.upper()}:")
                print(f"  Grid: {n_rows}x{n_cols}, Total plots: {total_plots}")
                print(f"  Titles: {subplot_titles}")

                # Calculate spacing with minimum thresholds
                vertical_spacing = max(0.05, min(0.15, 0.15 / n_rows)) if n_rows > 1 else 0.1
                horizontal_spacing = max(0.05, min(0.1, 0.1 / n_cols)) if n_cols > 1 else 0.1

                fig = make_subplots(
                    rows=n_rows,
                    cols=n_cols,
                    specs=specs,
                    subplot_titles=subplot_titles,
                    vertical_spacing=vertical_spacing,
                    horizontal_spacing=horizontal_spacing
                )

                # Add traces for each model and embedding type
                plot_index = 0
                for model_name, emb_types in embeddings_dict.items():
                    embeddings_to_process = []

                    if layer_mode == 'concatenate':
                        # Handle concatenated embeddings
                        if 'concatenated' in emb_types:
                            embeddings = emb_types['concatenated']
                            emb_type = 'concatenated'
                            embeddings_to_process = [(emb_type, embeddings)]

                            # Log embedding info
                            print(f"Processing {model_name} - Concatenated ({dim_name}):")
                            print(f"  Shape: {embeddings.shape}")
                            print(f"  Layers included: {emb_types.get('layers_included', [])}")
                            print(f"  Total dimension: {emb_types.get('total_dim', 'unknown')}")
                        else:
                            print(f"Warning: No concatenated embeddings for {model_name}")
                            plot_index += 1
                            continue
                    else:
                        # Original individual mode
                        for emb_type in sorted(emb_types.keys()):
                            embeddings = emb_types[emb_type]
                            embeddings_to_process.append((emb_type, embeddings))

                    # Process each embedding
                    for emb_type, embeddings in embeddings_to_process:
                        # Skip if empty
                        if len(embeddings) == 0:
                            print(f"Warning: Skipping empty embeddings for {model_name} - {emb_type}")
                            plot_index += 1
                            continue

                        # Log embedding shape for debugging
                        if layer_mode != 'concatenate':
                            print(f"Processing {model_name} - {emb_type} ({dim_name}): shape {embeddings.shape}")

                        # Reduce dimensions
                        try:
                            reduced_data, variance = self.reduce_dimensions(
                                embeddings, method, n_components
                            )
                            # Store variance for PCA (to potentially update titles later)
                            if variance is not None:
                                variance_key = f"{method}_{dim_name}_{plot_index}"
                                variance_info[variance_key] = variance
                        except ValueError as e:
                            print(f"Error reducing dimensions for {model_name} - {emb_type}: {e}")
                            plot_index += 1
                            continue

                        # Calculate subplot position
                        row = plot_index // n_cols + 1
                        col = plot_index % n_cols + 1

                        # Add trace for each word
                        for word_idx, word in enumerate(unique_words):
                            # Only show legend for first subplot
                            show_legend = (plot_index == 0)

                            trace = self.create_scatter_trace(
                                reduced_data, labels, color_map, word,
                                show_legend, dim_name
                            )

                            fig.add_trace(trace, row=row, col=col)

                        plot_index += 1

                # Update layout
                height = max(600, 400 * n_rows)
                width = max(1200, 450 * n_cols)

                layout_updates = {
                    'height': height,
                    'width': width,
                    'showlegend': True,
                    'title_text': f'{method.upper()} - {dim_name.upper()} Visualization',
                    'title_x': 0.5,
                    'title_font_size': 20
                }

                # Update axis labels for all subplots
                for i in range(1, plot_index + 1):
                    row = (i - 1) // n_cols + 1
                    col = (i - 1) % n_cols + 1

                    if dim_name == '3d':
                        scene_key = 'scene' if i == 1 else f'scene{i}'
                        # Set equal aspect ratio for 3D plots to preserve distances
                        layout_updates[scene_key] = dict(
                            xaxis=dict(
                                title='Component 1',
                                showgrid=True,
                                showbackground=True
                            ),
                            yaxis=dict(
                                title='Component 2',
                                showgrid=True,
                                showbackground=True
                            ),
                            zaxis=dict(
                                title='Component 3',
                                showgrid=True,
                                showbackground=True
                            ),
                            aspectmode='cube',  # Force equal aspect ratio on all axes
                            camera=dict(
                                eye=dict(x=1.5, y=1.5, z=1.5)
                            )
                        )
                    else:
                        xaxis_key = 'xaxis' if i == 1 else f'xaxis{i}'
                        yaxis_key = 'yaxis' if i == 1 else f'yaxis{i}'
                        # scaleanchor needs format 'y', 'y2', 'y3' (without 'axis')
                        scaleanchor_ref = 'y' if i == 1 else f'y{i}'
                        # Set equal aspect ratio for 2D plots to preserve distances (critical for t-SNE)
                        layout_updates[xaxis_key] = dict(
                            title='Component 1',
                            scaleanchor=scaleanchor_ref,  # Lock x-axis scale to y-axis
                            scaleratio=1,                 # 1:1 aspect ratio
                            constrain='domain'            # Constrain scaling to subplot domain
                        )
                        layout_updates[yaxis_key] = dict(
                            title='Component 2',
                            constrain='domain'
                        )

                fig.update_layout(**layout_updates)

                # Update subplot titles to include variance for PCA
                if method == 'pca' and variance_info:
                    for i in range(len(subplot_titles)):
                        variance_key = f"{method}_{dim_name}_{i}"
                        if variance_key in variance_info:
                            variance_pct = variance_info[variance_key] * 100
                            # Update annotation text
                            if i < len(fig.layout.annotations):
                                current_text = fig.layout.annotations[i].text
                                fig.layout.annotations[i].text = f"{current_text}<br><span style='font-size:10px'>Var: {variance_pct:.1f}%</span>"

                # Convert to HTML - use False to not include Plotly.js (already loaded in template)
                # Convert figure to JSON to avoid binary encoding issues
                import json
                fig_json = fig.to_json()

                # Create the HTML with proper Plotly initialization
                plot_html = f'''<div id="{method}_{dim_name}_plot" class="plotly-graph-div" style="height:{height}px; width:{width}px;"></div>
<script type="text/javascript">
    (function() {{
        var figure = {fig_json};
        var plotDiv = document.getElementById('{method}_{dim_name}_plot');
        if (plotDiv && window.Plotly) {{
            Plotly.newPlot(plotDiv, figure.data, figure.layout, {{"responsive": true}});
        }}
    }})();
</script>'''
                plots[f'{method}_{dim_name}'] = plot_html
                print(f"  Generated HTML: {len(plot_html)} chars")

                # Debug: print first 200 chars
                print(f"  Preview: {plot_html[:200]}...")

        if not plots:
            raise ValueError(
                "Could not generate any plots. Please ensure you have:\n"
                "1. Selected at least one model\n"
                "2. Selected at least one layer for each model\n"
                "3. Selected at least 2-3 words\n"
                "4. Set samples per word to at least 10\n"
                "Check the console for detailed error messages."
            )

        return plots