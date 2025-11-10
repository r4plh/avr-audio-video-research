// Global state
let modelInfo = {};
let currentCacheKey = null;

// Initialize on page load
$(document).ready(function() {
    initializeEventHandlers();
});

function initializeEventHandlers() {
    // Load model info button
    $('#load-model-info').click(loadModelInfo);

    // Word selection buttons
    $('#select-all-words').click(function() {
        $('.word-checkbox').prop('checked', true);
    });

    $('#deselect-all-words').click(function() {
        $('.word-checkbox').prop('checked', false);
    });

    // Pooling method change
    $('#pooling-method').change(function() {
        if ($(this).val() === 'position') {
            $('#pooling-position-container').show();
        } else {
            $('#pooling-position-container').hide();
        }
    });

    // Extract embeddings button
    $('#extract-embeddings').click(extractEmbeddings);

    // Visualize button
    $('#visualize-btn').click(generatePlots);

    // Clear cache button
    $('#clear-cache').click(clearCache);
}

function getSelectedModels() {
    let models = [];
    $('.model-checkbox:checked').each(function() {
        models.push($(this).val());
    });
    return models;
}

function getSelectedWords() {
    let words = [];
    $('.word-checkbox:checked').each(function() {
        words.push($(this).val());
    });
    return words;
}

function showStatus(message, type = 'info') {
    let alertClass = 'alert-' + type;
    $('#status-message')
        .removeClass('alert-info alert-success alert-warning alert-danger')
        .addClass(alertClass)
        .text(message);
}

function showProgress(show = true, title = 'Processing...', detail = 'Please wait...', percent = 0) {
    if (show) {
        $('#progress-container').fadeIn();
        $('#progress-title').text(title);
        $('#progress-detail').text(detail);

        // If percent is 0, show indeterminate progress
        if (percent === 0) {
            $('#progress-bar').css('width', '100%');
            $('#progress-text').text('Processing...');
        } else {
            $('#progress-bar').css('width', percent + '%');
            $('#progress-text').text(percent + '%');
        }
    } else {
        $('#progress-container').fadeOut();
        $('#progress-bar').css('width', '0%');
    }
}

function loadModelInfo() {
    let selectedModels = getSelectedModels();

    if (selectedModels.length === 0) {
        showStatus('Please select at least one model', 'warning');
        return;
    }

    showStatus('Loading model information...', 'info');
    showProgress(true, 'Loading Model Information', `Loading info for ${selectedModels.length} model(s)...`);

    $.ajax({
        url: '/get_model_info',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ models: selectedModels }),
        success: function(response) {
            modelInfo = response;
            renderLayerSelection(response);
            showStatus('Model info loaded successfully', 'success');
            showProgress(false);
        },
        error: function(xhr) {
            showStatus('Error loading model info: ' + xhr.responseText, 'danger');
            showProgress(false);
        }
    });
}

function renderLayerSelection(modelInfo) {
    let container = $('#layer-selection-container');
    container.empty();

    for (let modelName in modelInfo) {
        let info = modelInfo[modelName];

        if (info.error) {
            container.append(`
                <div class="alert alert-danger small">
                    ${modelName}: ${info.error}
                </div>
            `);
            continue;
        }

        let modelShort = modelName.split('/').pop();
        let modelDiv = $('<div class="mb-3 layer-selection-model"></div>');

        modelDiv.append(`<label class="form-label fw-bold small">${modelShort}:</label>`);

        // CNN option
        if (info.layers.cnn && info.layers.cnn.available) {
            modelDiv.append(`
                <div class="form-check">
                    <input class="form-check-input layer-checkbox"
                           type="checkbox"
                           data-model="${modelName}"
                           data-layer-type="cnn"
                           data-layer-index="0"
                           id="layer-${modelName}-cnn">
                    <label class="form-check-label small" for="layer-${modelName}-cnn">
                        CNN Features
                    </label>
                </div>
            `);
        }

        // Encoder layers
        if (info.layers.encoder && info.layers.encoder.available) {
            let encoderDiv = $('<div class="ms-3"></div>');
            encoderDiv.append('<label class="small text-muted">Encoder Layers:</label>');

            let layerCheckboxes = $('<div class="layer-checkboxes"></div>');
            for (let i of info.layers.encoder.layer_indices) {
                layerCheckboxes.append(`
                    <div class="form-check form-check-inline">
                        <input class="form-check-input layer-checkbox"
                               type="checkbox"
                               data-model="${modelName}"
                               data-layer-type="encoder"
                               data-layer-index="${i}"
                               id="layer-${modelName}-encoder-${i}">
                        <label class="form-check-label small" for="layer-${modelName}-encoder-${i}">
                            ${i}
                        </label>
                    </div>
                `);
            }
            encoderDiv.append(layerCheckboxes);
            modelDiv.append(encoderDiv);
        }

        // Decoder layers (Whisper only)
        if (info.layers.decoder && info.layers.decoder.available) {
            let decoderDiv = $('<div class="ms-3"></div>');
            decoderDiv.append('<label class="small text-muted">Decoder Layers:</label>');

            let layerCheckboxes = $('<div class="layer-checkboxes"></div>');
            for (let i of info.layers.decoder.layer_indices) {
                layerCheckboxes.append(`
                    <div class="form-check form-check-inline">
                        <input class="form-check-input layer-checkbox"
                               type="checkbox"
                               data-model="${modelName}"
                               data-layer-type="decoder"
                               data-layer-index="${i}"
                               id="layer-${modelName}-decoder-${i}">
                        <label class="form-check-label small" for="layer-${modelName}-decoder-${i}">
                            ${i}
                        </label>
                    </div>
                `);
            }
            decoderDiv.append(layerCheckboxes);
            modelDiv.append(decoderDiv);
        }

        container.append(modelDiv);
    }
}

function getLayerConfig() {
    let config = {};

    $('.layer-checkbox:checked').each(function() {
        let model = $(this).data('model');
        let layerType = $(this).data('layer-type');
        let layerIndex = $(this).data('layer-index');

        if (!config[model]) {
            config[model] = {};
        }

        if (layerType === 'cnn') {
            config[model]['cnn'] = true;
        } else {
            if (!config[model][layerType]) {
                config[model][layerType] = [];
            }
            config[model][layerType].push(layerIndex);
        }
    });

    return config;
}

function extractEmbeddings() {
    let selectedModels = getSelectedModels();
    let selectedWords = getSelectedWords();
    let layerConfig = getLayerConfig();

    // Validation
    if (selectedModels.length === 0) {
        showStatus('Please select at least one model', 'warning');
        return;
    }

    if (selectedWords.length === 0) {
        showStatus('Please select at least one word', 'warning');
        return;
    }

    if (Object.keys(layerConfig).length === 0) {
        showStatus('Please select at least one layer', 'warning');
        return;
    }

    let samplesPerWord = parseInt($('#samples-per-word').val());
    let poolingMethod = $('#pooling-method').val();
    let poolingPosition = parseInt($('#pooling-position').val());
    let layerMode = $('#layer-mode').val();

    let config = {
        models: selectedModels,
        words: selectedWords,
        samples_per_word: samplesPerWord,
        layer_configs: layerConfig,
        pooling_method: poolingMethod,
        pooling_position: poolingPosition,
        layer_mode: layerMode  // Add layer mode to config
    };

    // Calculate total operations for progress display
    let totalOperations = selectedModels.length * selectedWords.length * samplesPerWord;
    let detailMessage = `Processing ${totalOperations} embeddings (${selectedModels.length} models × ${selectedWords.length} words × ${samplesPerWord} samples)`;

    showStatus('Extracting embeddings... This may take a while.', 'info');
    showProgress(true, 'Extracting Embeddings', detailMessage);
    $('#extract-embeddings').prop('disabled', true);

    $.ajax({
        url: '/extract_embeddings',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(config),
        success: function(response) {
            if (response.status === 'success') {
                currentCacheKey = response.cache_key;
                showStatus(
                    `Embeddings extracted successfully! (${response.num_samples} samples)`,
                    'success'
                );
                $('#visualize-btn').prop('disabled', false);
            } else {
                showStatus('Error: ' + response.message, 'danger');
            }
            showProgress(false);
            $('#extract-embeddings').prop('disabled', false);
        },
        error: function(xhr) {
            let errorMsg = xhr.responseJSON ? xhr.responseJSON.message : xhr.responseText;
            showStatus('Error extracting embeddings: ' + errorMsg, 'danger');
            showProgress(false);
            $('#extract-embeddings').prop('disabled', false);
        }
    });
}

function generatePlots() {
    if (!currentCacheKey) {
        showStatus('Please extract embeddings first', 'warning');
        return;
    }

    let vizType = $('#viz-type').val();
    let dimensions = $('#dimensions').val();

    // Determine what we're generating
    let methodText = vizType === 'both' ? 'PCA and t-SNE' : vizType.toUpperCase();
    let dimText = dimensions === 'both' ? '2D and 3D' : dimensions.toUpperCase();

    showStatus('Generating plots...', 'info');
    showProgress(true, 'Generating Visualizations', `Creating ${methodText} plots in ${dimText} dimensions...`);
    $('#visualize-btn').prop('disabled', true);

    $.ajax({
        url: '/visualize',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            cache_key: currentCacheKey,
            viz_type: vizType,
            dimensions: dimensions
        }),
        success: function(response) {
            if (response.status === 'success') {
                renderPlots(response.plots);
                showStatus('Plots generated successfully!', 'success');
            } else {
                showStatus('Error: ' + response.message, 'danger');
            }
            showProgress(false);
            $('#visualize-btn').prop('disabled', false);
        },
        error: function(xhr) {
            let errorMsg = xhr.responseJSON ? xhr.responseJSON.message : xhr.responseText;
            showStatus('Error generating plots: ' + errorMsg, 'danger');
            showProgress(false);
            $('#visualize-btn').prop('disabled', false);
        }
    });
}

function renderPlots(plots) {
    console.log('=== renderPlots called ===');
    console.log('Plot keys:', Object.keys(plots));
    console.log('Number of plots:', Object.keys(plots).length);

    // Debug: log first 500 chars of each plot
    for (let key in plots) {
        console.log(`${key}: ${plots[key].substring(0, 500)}...`);
    }

    let container = $('#plots-container');
    container.empty();

    // Group plots by dimension and method
    let groupedPlots = {
        'pca_2d': null,
        'pca_3d': null,
        'tsne_2d': null,
        'tsne_3d': null
    };

    for (let plotName in plots) {
        groupedPlots[plotName] = plots[plotName];
        console.log(`  ${plotName}: ${plots[plotName].length} chars`);
    }

    // Render plots in organized sections
    let methods = ['pca', 'tsne'];
    let dimensions = ['2d', '3d'];

    for (let method of methods) {
        let methodHasPlots = false;

        // Check if this method has any plots
        for (let dim of dimensions) {
            if (groupedPlots[`${method}_${dim}`]) {
                methodHasPlots = true;
                break;
            }
        }

        if (!methodHasPlots) continue;

        // Add method header
        let methodHeader = $(`<h3 class="mt-4 mb-3">${method.toUpperCase()} Visualization</h3>`);
        container.append(methodHeader);

        // Add plots for each dimension
        for (let dim of dimensions) {
            let plotKey = `${method}_${dim}`;
            if (groupedPlots[plotKey]) {
                // Add dimension subheader
                let dimHeader = $(`<div class="plot-dimension-header">
                    <i class="fas fa-chart-${dim === '2d' ? 'area' : 'cube'}"></i>
                    ${dim.toUpperCase()} Plots
                </div>`);
                container.append(dimHeader);

                // Add plot - directly insert HTML
                let plotHtml = groupedPlots[plotKey];
                console.log(`Rendering ${plotKey}, HTML length: ${plotHtml.length}`);

                let plotDiv = $('<div class="plot-section"></div>');
                container.append(plotDiv);

                // Insert HTML and execute any scripts
                $(plotDiv).html(plotHtml);

                // Execute any script tags that were inserted
                $(plotDiv).find('script').each(function() {
                    eval($(this).text());
                });

                console.log(`  Appended HTML and executed scripts for ${plotKey}`);
            }
        }
    }

    // Scroll to plots
    $('html, body').animate({
        scrollTop: $('#plots-container').offset().top - 100
    }, 500);
}

function clearCache() {
    if (confirm('Are you sure you want to clear the cache?')) {
        showStatus('Clearing cache...', 'info');

        $.ajax({
            url: '/clear_cache',
            method: 'POST',
            success: function(response) {
                if (response.status === 'success') {
                    showStatus('Cache cleared successfully', 'success');
                    currentCacheKey = null;
                    $('#visualize-btn').prop('disabled', true);
                } else {
                    showStatus('Error clearing cache: ' + response.message, 'danger');
                }
            },
            error: function(xhr) {
                showStatus('Error clearing cache: ' + xhr.responseText, 'danger');
            }
        });
    }
}
