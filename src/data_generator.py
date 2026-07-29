import pandas as pd
import numpy as np
import yaml
import os
import random

def generate_data(num_samples: int, output_path: str):
    np.random.seed(42)
    random.seed(42)
    
    topics = [
        "Product Quality",
        "Customer Service",
        "Delivery & Shipping"
    ]
    
    templates = {
        "Product Quality": [
            "The product is amazing, highly recommend!",
            "Terrible quality, broke after one use.",
            "Decent item for the price, but feels a bit cheap.",
            "Absolutely love the design and build quality."
        ],
        "Customer Service": [
            "Customer support was very helpful and resolved my issue.",
            "Worst service ever, they never replied to my emails.",
            "The representative was polite but couldn't fix the problem.",
            "Five stars for the support team, they are great."
        ],
        "Delivery & Shipping": [
            "Fast shipping, arrived a day early!",
            "Package was delayed by a week, very disappointed.",
            "The box was damaged during delivery.",
            "Smooth delivery process, no complaints."
        ]
    }
    
    reviews = []
    labels = []
    
    for _ in range(num_samples):
        topic = random.choice(topics)
        review = random.choice(templates[topic])
        
        # Add some noise/variation
        if random.random() > 0.8:
            review = review.lower()
        if random.random() > 0.9:
            review += " " + random.choice(["Thanks", "Bye", "Whatever", "Ok"])
            
        reviews.append(review)
        labels.append(topic)

    df = pd.DataFrame({
        'review': reviews,
        'true_topic': labels
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_samples} reviews and saved to {output_path}")

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    generate_data(config['data']['num_samples'], config['data']['raw_path'])
