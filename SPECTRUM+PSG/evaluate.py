
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import base64
from io import BytesIO
import json

# Create output directory
os.makedirs("dashboard", exist_ok=True)
os.makedirs("dashboard/images", exist_ok=True)
os.makedirs("dashboard/data", exist_ok=True)

# Load model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()

# Your text examples
texts = [
    "This movie was fantastic! I really enjoyed it.",
    "The acting was terrible and the plot made no sense.",
    "The cinematography was beautiful but the story was confusing."
]

# Function to get prediction
def get_prediction(text):
    inputs = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()[0]

# Function to calculate token importance manually
def calculate_token_importance(text):
    # Get the baseline prediction for the full text
    baseline_pred = get_prediction(text)
    print(f"Baseline prediction (positive sentiment): {baseline_pred[1]:.4f}")
    
    # Tokenize the text properly
    encoded = tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
    offsets = encoded['offset_mapping']
    
    print(f"Found {len(tokens)} tokens: {tokens}")
    
    # Calculate importance for each token
    token_importance = []
    
    for i, (token, offset) in enumerate(zip(tokens, offsets)):
        # Create a version of the text with this token masked
        start, end = offset
        masked_text = text[:start] + tokenizer.mask_token + text[end:]
        
        # Get prediction for masked text
        masked_pred = get_prediction(masked_text)
        
        # Calculate importance as absolute difference in prediction
        importance = abs(baseline_pred[1] - masked_pred[1])
        token_importance.append((token, importance, start, end))
        
        print(f"Token: '{token}', Position: {start}-{end}, Importance: {importance:.6f}")
    
    return token_importance, tokens, baseline_pred

# Function to convert matplotlib figure to base64 string for HTML embedding
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

# Function to create a more visible colored text visualization
def create_visible_token_visualization(token_importance, text_number):
    # Sort by position in text
    token_importance.sort(key=lambda x: x[2])
    
    # Create a larger figure with white background
    plt.figure(figsize=(20, 6), facecolor='white')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    
    # Create a white background
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Calculate maximum importance for color scaling
    max_importance = max([t[1] for t in token_importance])
    
    # Set up text positioning
    x_position = 0.05
    y_position = 0.5
    line_height = 0.15
    max_line_width = 0.9  # As a fraction of figure width
    
    # Track current line width
    current_line_width = 0
    
    # Add tokens with enhanced visibility
    for token, importance, start, end in token_importance:
        # Skip empty tokens
        if not token.strip():
            continue
            
        # Calculate relative importance (0-1)
        rel_importance = importance / max_importance
        
        # Create color with minimum intensity to ensure visibility
        # Red for high importance, pink for low importance
        r = 1.0
        g = max(0.0, min(0.9, 1.0 - rel_importance))
        b = max(0.0, min(0.9, 1.0 - rel_importance))
        
        # Calculate token width (approximate)
        token_width = len(token) * 0.01
        
        # Check if we need to move to next line
        if current_line_width + token_width > max_line_width:
            x_position = 0.05
            y_position -= line_height
            current_line_width = 0
        
        # Draw a light background behind the text for better contrast
        text_box = plt.Rectangle((x_position-0.005, y_position-0.06), 
                                token_width+0.01, 0.12, 
                                color='#f8f8f8', alpha=0.7, 
                                transform=ax.transAxes)
        ax.add_patch(text_box)
        
        # Add text with black outline for maximum visibility
        text_obj = plt.text(x_position, y_position, token, 
                          color=(r, g, b), 
                          fontsize=16, 
                          fontweight='bold',
                          transform=ax.transAxes)
        
        # Add black outline to text
        text_obj.set_path_effects([
            PathEffects.Stroke(linewidth=2, foreground='black'),
            PathEffects.Normal()
        ])
        
        # Update position for next token
        x_position += token_width + 0.01  # Add space between tokens
        current_line_width += token_width + 0.01
        
        # Add extra space after sentence-ending punctuation
        if token in ['.', '!', '?']:
            x_position += 0.01
            current_line_width += 0.01
    
    # Remove axes
    plt.axis('off')
    
    # Save high-resolution image
    file_path = f"dashboard/images/colored_text_{text_number}.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    
    # Get base64 encoding for HTML embedding
    img_str = fig_to_base64(plt.gcf())
    plt.close()
    
    return img_str, file_path

# Process each text with manual approach
all_results = []

for i, text in enumerate(texts):
    print(f"\n\nAnalyzing text {i+1}: {text}")
    
    # Calculate token importance
    token_importance, tokens, baseline_pred = calculate_token_importance(text)
    
    # Save token importance data as JSON
    token_data = []
    for token, importance, start, end in token_importance:
        token_data.append({
            "token": token,
            "importance": float(importance),
            "position": [int(start), int(end)]
        })
    
    with open(f"dashboard/data/token_data_{i+1}.json", "w") as f:
        json.dump(token_data, f)
    
    # Sort by importance for visualizations
    token_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Create visualization of top 10 tokens
    plt.figure(figsize=(12, 5))
    top_tokens = token_importance[:10]
    plt.bar([t[0] for t in top_tokens], [t[1] for t in top_tokens], color='#3498db')
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Top 10 Most Important Tokens", fontsize=14)
    plt.tight_layout()
    top_tokens_path = f"dashboard/images/top_tokens_{i+1}.png"
    plt.savefig(top_tokens_path)
    top_tokens_img = fig_to_base64(plt.gcf())
    plt.close()
    
    # Generate the enhanced colored text visualization
    colored_text_img, colored_text_path = create_visible_token_visualization(token_importance, i+1)
    
    # Create sentiment visualization
    plt.figure(figsize=(8, 4))
    sentiment_scores = [baseline_pred[0], baseline_pred[1]]
    plt.bar(['Negative', 'Positive'], sentiment_scores, color=['#e74c3c', '#2ecc71'])
    plt.ylim(0, 1)
    plt.title("Sentiment Analysis", fontsize=14)
    for j, v in enumerate(sentiment_scores):
        plt.text(j, v + 0.01, f"{v:.2f}", ha='center', fontsize=12)
    plt.tight_layout()
    sentiment_path = f"dashboard/images/sentiment_{i+1}.png"
    plt.savefig(sentiment_path)
    sentiment_img = fig_to_base64(plt.gcf())
    plt.close()
    
    # Store results for dashboard
    all_results.append({
        "id": i+1,
        "text": text,
        "sentiment": {
            "positive": float(baseline_pred[1]),
            "negative": float(baseline_pred[0])
        },
        "token_count": len(tokens),
        "top_tokens": [{"token": t[0], "importance": float(t[1])} for t in top_tokens],
        "images": {
            "colored_text": colored_text_path,
            "top_tokens": top_tokens_path,
            "sentiment": sentiment_path
        }
    })
dashboard_html=dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Token Importance Analysis Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            padding: 20px;
        }
        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .card {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: none;
        }
        .card-header {
            background-color: #4a6fa5;
            color: white;
            border-radius: 10px 10px 0 0 !important;
            font-weight: 600;
        }
        .nav-pills .nav-link.active {
            background-color: #4a6fa5;
        }
        .nav-pills .nav-link {
            color: #4a6fa5;
        }
        .text-display {
            font-size: 18px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .sentiment-badge {
            font-size: 14px;
            padding: 5px 10px;
            border-radius: 15px;
        }
        .token-table {
            max-height: 300px;
            overflow-y: auto;
        }
        .visualization-container {
            text-align: center;
            margin: 15px 0;
        }
        .visualization-container img {
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .dashboard-header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .summary-card {
            height: 100%;
        }
        .token-importance-chart {
            height: 300px;
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="dashboard-header">
            <h1 class="mb-0">Token Importance Analysis Dashboard</h1>
            <p class="mb-0 mt-2">Visualizing how individual tokens impact sentiment predictions</p>
        </div>
        
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">Text Selection</div>
                    <div class="card-body">
                        <ul class="nav nav-pills mb-3" id="text-tabs" role="tablist">
                            <!-- Tabs will be generated by JavaScript -->
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="tab-content" id="text-content">
            <!-- Tab content will be generated by JavaScript -->
        </div>
    </div>

    <script>
        // Load the analysis results
        fetch('data/all_results.json')
            .then(response => response.json())
            .then(data => {
                // Generate tabs
                const tabsContainer = document.getElementById('text-tabs');
                data.forEach((result, index) => {
                    const isActive = index === 0 ? 'active' : '';
                    tabsContainer.innerHTML += `
                        <li class="nav-item" role="presentation">
                            <button class="nav-link ${isActive}" id="text-${result.id}-tab" data-bs-toggle="pill" 
                                    data-bs-target="#text-${result.id}" type="button" role="tab" 
                                    aria-controls="text-${result.id}" aria-selected="${index === 0}">
                                Text ${result.id}
                            </button>
                        </li>
                    `;
                });
                
                // Generate tab content
                const contentContainer = document.getElementById('text-content');
                data.forEach((result, index) => {
                    const isActive = index === 0 ? 'show active' : '';
                    const sentimentClass = result.sentiment.positive > 0.5 ? 'bg-success' : 'bg-danger';
                    const sentimentText = result.sentiment.positive > 0.5 ? 'Positive' : 'Negative';
                    
                    contentContainer.innerHTML += `
                        <div class="tab-pane fade ${isActive}" id="text-${result.id}" role="tabpanel" 
                             aria-labelledby="text-${result.id}-tab">
                            
                            <div class="row mb-4">
                                <div class="col-md-8">
                                    <div class="card">
                                        <div class="card-header">Original Text</div>
                                        <div class="card-body">
                                            <div class="text-display">"${result.text}"</div>
                                            <div>
                                                <span class="sentiment-badge ${sentimentClass} text-white">
                                                    ${sentimentText} (${(result.sentiment.positive * 100).toFixed(1)}%)
                                                </span>
                                                <span class="ms-2">Total Tokens: ${result.token_count}</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="card summary-card">
                                        <div class="card-header">Sentiment Analysis</div>
                                        <div class="card-body">
                                            <div class="visualization-container">
                                                <img src="${result.images.sentiment}" alt="Sentiment Analysis">
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row mb-4">
                                <div class="col-12">
                                    <div class="card">
                                        <div class="card-header">Token Importance Visualization</div>
                                        <div class="card-body">
                                            <div class="visualization-container">
                                                <img src="${result.images.colored_text}" alt="Token Importance">
                                            </div>
                                            <p class="text-center text-muted mt-2">
                                                Color intensity indicates token importance (darker red = more important)
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row">
                                <div class="col-md-7">
                                    <div class="card">
                                        <div class="card-header">Top 10 Most Important Tokens</div>
                                        <div class="card-body">
                                            <div class="visualization-container">
                                                <img src="${result.images.top_tokens}" alt="Top Tokens">
                                            </div>
                                            <div class="mt-4">
                                                <canvas id="token-importance-chart-${result.id}" class="token-importance-chart"></canvas>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-5">
                                    <div class="card">
                                        <div class="card-header">Token Importance Details</div>
                                        <div class="card-body">
                                            <div class="token-table">
                                                <table class="table table-striped">
                                                    <thead>
                                                        <tr>
                                                            <th>Token</th>
                                                            <th>Importance</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody id="token-table-${result.id}">
                                                        <!-- Will be populated by JavaScript -->
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // Load token data for this text
                    fetch(`data/token_data_${result.id}.json`)
                        .then(response => response.json())
                        .then(tokenData => {
                            // Sort by importance
                            tokenData.sort((a, b) => b.importance - a.importance);
                            
                            // Populate token table
                            const tableBody = document.getElementById(`token-table-${result.id}`);
                            tokenData.forEach(token => {
                                tableBody.innerHTML += `
                                    <tr>
                                        <td>${token.token}</td>
                                        <td>${token.importance.toFixed(6)}</td>
                                    </tr>
                                `;
                            });
                            
                            // Create token importance chart
                            const ctx = document.getElementById(`token-importance-chart-${result.id}`);
                            new Chart(ctx, {
                                type: 'bar',
                                data: {
                                    labels: tokenData.slice(0, 10).map(t => t.token),
                                    datasets: [{
                                        label: 'Token Importance',
                                        data: tokenData.slice(0, 10).map(t => t.importance),
                                        backgroundColor: '#3498db'
                                    }]
                                },
                                options: {
                                    responsive: true,
                                    maintainAspectRatio: false,
                                    scales: {
                                        y: {
                                            beginAtZero: true
                                        }
                                    }
                                }
                            });
                        })
                        .catch(error => console.error('Error loading token data:', error));
                });
            })
            .catch(error => console.error('Error loading results:', error));
    </script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# Create the dashboard HTML file
with open("dashboard/index.html", "w", encoding="utf-8") as f:
    f.write(dashboard_html)

print("\nAnalysis complete. Dashboard created in 'dashboard' directory.")
print("Open 'dashboard/index.html' in a web browser to view the dashboard.")
