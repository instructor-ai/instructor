---
title: Example Browser
description: Browse and filter Instructor examples by category, provider, and difficulty
tags: [cookbook, examples, browser]
---

# Example Browser

Find the perfect example for your use case by filtering by category, provider, and difficulty level.

<div class="example-browser">
    <div class="filters">
        <div class="filter-group">
            <label for="category-filter">Category:</label>
            <select id="category-filter">
                <option value="">All Categories</option>
                <option value="classification">Classification</option>
                <option value="extraction">Information Extraction</option>
                <option value="document">Document Processing</option>
                <option value="vision">Vision & Multimodal</option>
                <option value="database">Database Integration</option>
                <option value="streaming">Streaming & Processing</option>
            </select>
        </div>
        
        <div class="filter-group">
            <label for="provider-filter">Provider:</label>
            <select id="provider-filter">
                <option value="">All Providers</option>
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
                <option value="gemini">Google Gemini</option>
                <option value="mistral">Mistral</option>
                <option value="ollama">Ollama</option>
                <option value="llama-cpp">llama-cpp-python</option>
                <option value="groq">Groq</option>
                <option value="cohere">Cohere</option>
            </select>
        </div>
        
        <div class="filter-group">
            <label for="difficulty-filter">Difficulty:</label>
            <select id="difficulty-filter">
                <option value="">All Difficulties</option>
                <option value="beginner">Beginner</option>
                <option value="intermediate">Intermediate</option>
                <option value="advanced">Advanced</option>
            </select>
        </div>

        <button id="reset-filters" class="md-button">Reset Filters</button>
    </div>
    
    <div id="example-count" class="example-count">
        Showing 0 examples
    </div>
    
    <div class="results" id="example-results">
        <div class="loader">Loading examples...</div>
    </div>
</div>

<style>
.example-browser {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: var(--md-code-bg-color);
    margin-bottom: 2rem;
}

.filters {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 1.5rem;
    align-items: flex-end;
}

.filter-group {
    display: flex;
    flex-direction: column;
    min-width: 200px;
}

.filter-group label {
    font-size: 0.8rem;
    margin-bottom: 0.3rem;
    font-weight: bold;
}

.filter-group select {
    padding: 0.5rem;
    border-radius: 0.25rem;
    border: 1px solid var(--md-typeset-color);
    background-color: var(--md-default-bg-color);
    color: var(--md-typeset-color);
}

.example-count {
    margin-bottom: 1rem;
    font-size: 0.9rem;
    color: var(--md-default-fg-color--light);
}

.example-card {
    border: 1px solid var(--md-default-fg-color--lightest);
    border-radius: 0.5rem;
    padding: 1rem;
    margin-bottom: 1rem;
    background-color: var(--md-default-bg-color);
    transition: transform 0.2s, box-shadow 0.2s;
}

.example-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.example-card h3 {
    margin-top: 0;
    margin-bottom: 0.5rem;
}

.example-card p {
    margin-bottom: 0.5rem;
}

.example-card .example-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1rem;
}

.example-card .tag {
    font-size: 0.7rem;
    padding: 0.2rem 0.5rem;
    border-radius: 1rem;
    background-color: var(--md-code-bg-color);
}

.example-card .tag.provider {
    background-color: #e2f0fb;
    color: #004085;
}

.example-card .tag.category {
    background-color: #d4edda;
    color: #155724;
}

.example-card .tag.difficulty {
    background-color: #fff3cd;
    color: #856404;
}

.example-card .tag.difficulty.beginner {
    background-color: #d4edda;
    color: #155724;
}

.example-card .tag.difficulty.intermediate {
    background-color: #fff3cd;
    color: #856404;
}

.example-card .tag.difficulty.advanced {
    background-color: #f8d7da;
    color: #721c24;
}

.loader {
    text-align: center;
    padding: 2rem;
    color: var(--md-default-fg-color--light);
}

#reset-filters {
    padding: 0.5rem 1rem;
    background-color: var(--md-default-fg-color--lightest);
    color: var(--md-default-fg-color);
    border: none;
    border-radius: 0.25rem;
    cursor: pointer;
}

#reset-filters:hover {
    background-color: var(--md-default-fg-color--light);
    color: var(--md-default-bg-color);
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
    // Example data - this would be generated from actual docs
    const examples = [
        {
            title: "Single Classification",
            url: "single_classification.md",
            description: "Basic classification with a single category",
            categories: ["classification"],
            providers: ["openai", "anthropic"],
            difficulty: "beginner"
        },
        {
            title: "Multiple Classification",
            url: "multiple_classification.md",
            description: "Handling multiple classification categories",
            categories: ["classification"],
            providers: ["openai", "gemini"],
            difficulty: "intermediate"
        },
        {
            title: "Entity Resolution",
            url: "entity_resolution.md",
            description: "Identify and disambiguate entities",
            categories: ["extraction"],
            providers: ["openai", "anthropic"],
            difficulty: "intermediate"
        },
        {
            title: "Contact Information",
            url: "extract_contact_info.md",
            description: "Extract structured contact details",
            categories: ["extraction"],
            providers: ["openai"],
            difficulty: "beginner"
        },
        {
            title: "Table Extraction",
            url: "tables_from_vision.md",
            description: "Convert image tables to structured data",
            categories: ["vision", "extraction"],
            providers: ["openai"],
            difficulty: "advanced"
        },
        {
            title: "Document Segmentation",
            url: "document_segmentation.md",
            description: "Divide documents into meaningful sections",
            categories: ["document"],
            providers: ["openai", "anthropic"],
            difficulty: "intermediate"
        },
        {
            title: "Knowledge Graph Generation",
            url: "knowledge_graph.md",
            description: "Create relationship graphs from text",
            categories: ["document", "extraction"],
            providers: ["openai", "groq"],
            difficulty: "advanced"
        },
        {
            title: "SQLModel Integration",
            url: "sqlmodel.md",
            description: "Store AI-generated data in SQL databases",
            categories: ["database"],
            providers: ["openai"],
            difficulty: "intermediate"
        },
        {
            title: "Pandas DataFrame",
            url: "pandas_df.md",
            description: "Work with structured data in Pandas",
            categories: ["database"],
            providers: ["openai"],
            difficulty: "intermediate"
        },
        {
            title: "Partial Response Streaming",
            url: "partial_streaming.md",
            description: "Stream partial results in real-time",
            categories: ["streaming"],
            providers: ["openai", "anthropic"],
            difficulty: "advanced"
        },
        {
            title: "Groq Integration",
            url: "groq.md",
            description: "High-performance inference with Groq",
            categories: ["classification", "extraction"],
            providers: ["groq"],
            difficulty: "beginner"
        },
        {
            title: "Mistral Integration",
            url: "mistral.md",
            description: "Using Mistral/Mixtral models",
            categories: ["classification"],
            providers: ["mistral"],
            difficulty: "beginner"
        },
        {
            title: "Ollama Integration",
            url: "ollama.md",
            description: "Local deployment with Ollama",
            categories: ["extraction"],
            providers: ["ollama"],
            difficulty: "intermediate"
        },
        {
            title: "Image to Ad Copy",
            url: "image_to_ad_copy.md",
            description: "Generate ad text from images",
            categories: ["vision"],
            providers: ["openai", "gemini"],
            difficulty: "intermediate"
        },
        {
            title: "YouTube Clip Analysis",
            url: "youtube_clips.md",
            description: "Extract info from video clips",
            categories: ["vision", "extraction"],
            providers: ["openai"],
            difficulty: "advanced"
        }
    ];

    const categoryFilter = document.getElementById('category-filter');
    const providerFilter = document.getElementById('provider-filter');
    const difficultyFilter = document.getElementById('difficulty-filter');
    const resetButton = document.getElementById('reset-filters');
    const resultsContainer = document.getElementById('example-results');
    const exampleCountElement = document.getElementById('example-count');

    function renderExamples() {
        const selectedCategory = categoryFilter.value;
        const selectedProvider = providerFilter.value;
        const selectedDifficulty = difficultyFilter.value;

        const filteredExamples = examples.filter(example => {
            // Apply category filter
            if (selectedCategory && !example.categories.includes(selectedCategory)) {
                return false;
            }
            
            // Apply provider filter
            if (selectedProvider && !example.providers.includes(selectedProvider)) {
                return false;
            }
            
            // Apply difficulty filter
            if (selectedDifficulty && example.difficulty !== selectedDifficulty) {
                return false;
            }
            
            return true;
        });

        // Update count
        exampleCountElement.textContent = `Showing ${filteredExamples.length} of ${examples.length} examples`;
        
        // Clear previous results
        resultsContainer.innerHTML = '';
        
        if (filteredExamples.length === 0) {
            resultsContainer.innerHTML = '<div class="loader">No examples match your filters</div>';
            return;
        }
        
        // Render results
        filteredExamples.forEach(example => {
            const card = document.createElement('div');
            card.className = 'example-card';
            
            const content = `
                <h3><a href="${example.url}">${example.title}</a></h3>
                <p>${example.description}</p>
                <div class="example-tags">
                    ${example.categories.map(cat => `<span class="tag category">${cat}</span>`).join('')}
                    ${example.providers.map(provider => `<span class="tag provider">${provider}</span>`).join('')}
                    <span class="tag difficulty ${example.difficulty}">${example.difficulty}</span>
                </div>
            `;
            
            card.innerHTML = content;
            resultsContainer.appendChild(card);
        });
    }

    // Event listeners
    categoryFilter.addEventListener('change', renderExamples);
    providerFilter.addEventListener('change', renderExamples);
    difficultyFilter.addEventListener('change', renderExamples);
    
    resetButton.addEventListener('click', function() {
        categoryFilter.value = '';
        providerFilter.value = '';
        difficultyFilter.value = '';
        renderExamples();
    });
    
    // Initial render
    renderExamples();
});
</script>

## Why Use the Example Browser?

The Example Browser helps you quickly find relevant examples based on your:

- **Use case category** - Find examples relevant to your task
- **Provider preference** - See examples using your chosen LLM provider
- **Experience level** - Start with beginner examples or jump to advanced techniques

Each example includes links to the full documentation page with complete code samples and explanations.

## Example Categories

Our examples are organized into these main categories:

- **Classification** - Sort and categorize content
- **Information Extraction** - Pull structured data from text
- **Document Processing** - Handle and analyze documents
- **Vision & Multimodal** - Work with images and other media
- **Database Integration** - Store and query AI-generated data
- **Streaming & Processing** - Handle streaming responses

Can't find what you need? Check our [full examples list](index.md) or ask in our [Discord community](https://discord.gg/bD9YE9JArw).