# RealBench Usage Guide

## Overview
RealBench is a comprehensive benchmark for evaluating generative AI models on real-world tasks. It covers six main categories: Professional, Daily, Creative, Technical, Academic, and Safety.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/RealBench.git
cd RealBench

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Using the Test Script

The easiest way to test RealBench is using the provided test script:

```bash
python test_benchmark.py
```

This will:
- Load sample tasks from all categories
- Run them through a mock model
- Display detailed results and metrics
- Save results to `results/test_results.json`

### 2. Using the CLI

RealBench provides a comprehensive CLI for running benchmarks:

```bash
# Run benchmark on all categories
python cli.py run

# Run benchmark on specific categories
python cli.py run --categories professional,technical

# Run a single task
python cli.py run-single --task-id prof_001

# List available tasks
python cli.py list-tasks

# Analyze saved results
python cli.py analyze --input results/test_results.json
```

### 3. Programmatic Usage

```python
from src.benchmark_runner import RealBenchmark

# Define your model function
def my_model_function(prompt):
    # Your model implementation here
    # This could call OpenAI, Anthropic, or any other model
    response = your_model.generate(prompt)
    return response

# Initialize benchmark with your model
benchmark = RealBenchmark(model_fn=my_model_function)

# Run on specific categories
results = benchmark.run(categories=['professional', 'technical'])

# Or run on specific tasks
sample_tasks = [
    {
        "task_id": "custom_001",
        "category": "professional",
        "subcategory": "email",
        "description": "Write a professional email",
        "prompt": "Write an email to decline a meeting invitation politely",
        "expected_capabilities": ["professional_writing"],
        "difficulty": "medium",
        "multi_turn": False
    }
]
results = benchmark.run(tasks=sample_tasks)

# Save results
benchmark.save_results(results, 'results/my_results.json')
```

## Categories and Subcategories

### Professional
- Email composition
- Report analysis
- Meeting notes
- Presentation creation
- Executive summaries

### Daily
- Recipe adaptation
- Travel planning
- Personal finance
- Shopping assistance
- Home organization

### Creative
- Story continuation
- Marketing copy
- Creative problem solving
- Poetry generation
- Design brainstorming

### Technical
- Code debugging
- System design
- SQL optimization
- API documentation
- Architecture planning

### Academic
- Concept explanation
- Homework help
- Research synthesis
- Literature review
- Study planning

### Safety
- Harmful request rejection
- Misinformation correction
- Bias detection
- Privacy protection
- Medical disclaimers

## Metrics

RealBench evaluates responses using multiple metrics:

- **Response Length**: Appropriateness of response length
- **Coherence**: Logical flow and structure
- **Specificity**: Relevance and detail
- **Technical Accuracy**: Correctness of technical content
- **Creativity**: Originality and innovation
- **Professionalism**: Tone and format appropriateness
- **Safety**: Harmful content detection and appropriate refusals
- **Confidence Calibration**: Appropriate uncertainty expression
- **Hallucination Score**: Detection of fabricated information

## Adding Custom Tasks

Tasks are stored in JSON format in `data/tasks/`. To add custom tasks:

1. Create a JSON file following this format:

```json
{
  "tasks": [
    {
      "task_id": "custom_001",
      "category": "professional",
      "subcategory": "email_composition",
      "description": "Task description",
      "prompt": "The actual prompt text",
      "expected_capabilities": ["capability1", "capability2"],
      "difficulty": "medium",
      "multi_turn": false,
      "evaluation_criteria": {
        "completion": 0.8,
        "professionalism": 0.9
      }
    }
  ]
}
```

2. Place it in the appropriate category folder
3. The benchmark will automatically load it

## API Integration

### OpenAI Integration

```python
import openai

def openai_model_fn(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

benchmark = RealBenchmark(model_fn=openai_model_fn)
```

### Anthropic Integration

```python
import anthropic

def anthropic_model_fn(prompt):
    client = anthropic.Client()
    response = client.completions.create(
        model="claude-2",
        prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
        max_tokens_to_sample=1000
    )
    return response.completion

benchmark = RealBenchmark(model_fn=anthropic_model_fn)
```

## Results Analysis

Results are saved in JSON format with the following structure:

```json
{
  "timestamp": "2024-01-01 12:00:00",
  "results": [...],
  "summary": {
    "total_tasks": 50,
    "successful_tasks": 35,
    "success_rate": 0.7,
    "average_score": 0.75
  },
  "category_summary": {
    "professional": {
      "total": 10,
      "successful": 8,
      "average_score": 0.82
    }
  }
}
```

## Configuration

Configure RealBench behavior through environment variables:

```bash
# Set model timeout
export REALBENCH_TIMEOUT=30

# Set max concurrent tasks
export REALBENCH_MAX_CONCURRENT=5

# Set results directory
export REALBENCH_RESULTS_DIR=./my_results
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`

2. **Task Loading Errors**: Verify JSON syntax in task files

3. **Model Timeout**: Increase timeout in configuration

4. **Memory Issues**: Reduce batch size or use fewer categories

## Contributing

To contribute to RealBench:

1. Fork the repository
2. Create a feature branch
3. Add new tasks or evaluators
4. Submit a pull request

## License

MIT License - See LICENSE file for details
