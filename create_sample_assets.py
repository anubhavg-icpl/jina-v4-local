#!/usr/bin/env python3
"""
Create sample images and assets for testing Jina Embeddings v4

This script creates various sample images with different content types
for testing the multimodal capabilities of Jina Embeddings v4.

Author: Claude
Date: 2025
"""

from PIL import Image, ImageDraw, ImageFont
import os
import random
import numpy as np


def create_gradient_background(width, height, color1, color2):
    """Create a gradient background"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    for i in range(height):
        ratio = i / height
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        draw.rectangle([0, i, width, i+1], fill=(r, g, b))
    
    return img


def create_sample_images():
    """Create various sample images for testing"""
    
    # Create directories
    os.makedirs("assets", exist_ok=True)
    os.makedirs("assets/samples", exist_ok=True)
    
    print("🎨 Creating sample images...")
    
    # Sample 1: Text-based image with technology theme
    img1 = create_gradient_background(600, 400, (70, 130, 180), (25, 25, 112))
    draw1 = ImageDraw.Draw(img1)
    
    # Add shapes for decoration
    for _ in range(5):
        x = random.randint(50, 550)
        y = random.randint(50, 350)
        size = random.randint(20, 40)
        draw1.ellipse([x, y, x+size, y+size], outline='white', width=2)
    
    # Add text
    try:
        # Try to add text with larger size
        draw1.text((300, 100), "Artificial Intelligence", fill='white', anchor="mm")
        draw1.text((300, 150), "Machine Learning", fill='lightblue', anchor="mm")
        draw1.text((300, 200), "Deep Neural Networks", fill='cyan', anchor="mm")
        draw1.text((300, 250), "Computer Vision", fill='lightgreen', anchor="mm")
        draw1.text((300, 300), "Natural Language Processing", fill='yellow', anchor="mm")
    except:
        # Fallback text placement
        draw1.text((100, 100), "Artificial Intelligence", fill='white')
        draw1.text((100, 150), "Machine Learning", fill='lightblue')
        draw1.text((100, 200), "Deep Neural Networks", fill='cyan')
        draw1.text((100, 250), "Computer Vision", fill='lightgreen')
        draw1.text((100, 300), "Natural Language Processing", fill='yellow')
    
    img1.save("assets/tech_concepts.png")
    print("   ✅ Created: tech_concepts.png")
    
    # Sample 2: Nature scene
    img2 = Image.new('RGB', (600, 400), color='skyblue')
    draw2 = ImageDraw.Draw(img2)
    
    # Draw sun
    draw2.ellipse([500, 50, 570, 120], fill='yellow', outline='orange', width=3)
    
    # Draw mountains
    points = [(0, 300), (150, 150), (300, 250), (450, 100), (600, 200), (600, 400), (0, 400)]
    draw2.polygon(points, fill='darkgreen')
    
    # Draw ground
    draw2.rectangle([0, 300, 600, 400], fill='green')
    
    # Draw trees
    for x in [100, 250, 400, 500]:
        # Tree trunk
        draw2.rectangle([x-10, 280, x+10, 320], fill='brown')
        # Tree leaves
        draw2.ellipse([x-30, 240, x+30, 300], fill='darkgreen')
    
    # Add text label
    draw2.text((300, 350), "Beautiful Nature Scene", fill='white', anchor="mm")
    
    img2.save("assets/nature_scene.png")
    print("   ✅ Created: nature_scene.png")
    
    # Sample 3: Abstract geometric art
    img3 = Image.new('RGB', (600, 400), color='black')
    draw3 = ImageDraw.Draw(img3)
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    
    # Draw random geometric shapes
    for _ in range(20):
        shape_type = random.choice(['rectangle', 'ellipse', 'polygon'])
        color = random.choice(colors)
        
        if shape_type == 'rectangle':
            x1, y1 = random.randint(0, 500), random.randint(0, 300)
            x2, y2 = x1 + random.randint(50, 100), y1 + random.randint(50, 100)
            draw3.rectangle([x1, y1, x2, y2], fill=color, outline='white')
        
        elif shape_type == 'ellipse':
            x1, y1 = random.randint(0, 500), random.randint(0, 300)
            x2, y2 = x1 + random.randint(50, 100), y1 + random.randint(50, 100)
            draw3.ellipse([x1, y1, x2, y2], fill=color, outline='white')
        
        else:  # polygon
            num_points = random.randint(3, 6)
            points = []
            center_x = random.randint(100, 500)
            center_y = random.randint(100, 300)
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                radius = random.randint(30, 60)
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                points.append((x, y))
            draw3.polygon(points, fill=color, outline='white')
    
    draw3.text((300, 370), "Abstract Geometric Art", fill='white', anchor="mm")
    
    img3.save("assets/abstract_art.png")
    print("   ✅ Created: abstract_art.png")
    
    # Sample 4: Data visualization mockup
    img4 = Image.new('RGB', (600, 400), color='white')
    draw4 = ImageDraw.Draw(img4)
    
    # Draw axes
    draw4.line([(50, 350), (550, 350)], fill='black', width=2)  # X-axis
    draw4.line([(50, 50), (50, 350)], fill='black', width=2)    # Y-axis
    
    # Draw bar chart
    bar_colors = ['blue', 'green', 'red', 'orange', 'purple']
    bar_heights = [250, 180, 300, 220, 150]
    bar_width = 80
    
    for i, (height, color) in enumerate(zip(bar_heights, bar_colors)):
        x = 100 + i * 100
        draw4.rectangle([x, 350-height, x+bar_width, 350], fill=color)
        draw4.text((x+bar_width//2, 370), f"Cat {i+1}", fill='black', anchor="mm")
    
    # Title
    draw4.text((300, 30), "Sample Data Visualization", fill='black', anchor="mm")
    
    img4.save("assets/data_viz.png")
    print("   ✅ Created: data_viz.png")
    
    # Sample 5: Simple icons
    for icon_name, emoji, bg_color, text in [
        ("icon_home.png", "🏠", (255, 230, 200), "Home"),
        ("icon_search.png", "🔍", (200, 230, 255), "Search"),
        ("icon_settings.png", "⚙️", (230, 230, 230), "Settings"),
        ("icon_user.png", "👤", (255, 200, 230), "User Profile"),
        ("icon_chart.png", "📊", (200, 255, 200), "Analytics")
    ]:
        img = Image.new('RGB', (200, 200), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw border
        draw.rectangle([10, 10, 190, 190], outline='gray', width=2)
        
        # Add text
        try:
            draw.text((100, 80), emoji, fill='black', anchor="mm")
            draw.text((100, 130), text, fill='black', anchor="mm")
        except:
            draw.text((50, 80), emoji, fill='black')
            draw.text((50, 130), text, fill='black')
        
        img.save(f"assets/{icon_name}")
        print(f"   ✅ Created: {icon_name}")
    
    # Sample 6: Text documents as images
    documents = [
        ("doc_python.png", "Python Programming", [
            "def hello_world():",
            "    print('Hello, World!')",
            "    return True",
            "",
            "# Main execution",
            "if __name__ == '__main__':",
            "    hello_world()"
        ]),
        ("doc_ml.png", "Machine Learning", [
            "import numpy as np",
            "from sklearn import svm",
            "",
            "# Train SVM classifier",
            "X = [[0, 0], [1, 1]]",
            "y = [0, 1]",
            "clf = svm.SVC()",
            "clf.fit(X, y)"
        ]),
        ("doc_readme.png", "README", [
            "# Project Title",
            "",
            "## Description",
            "This is a sample project",
            "",
            "## Installation",
            "pip install -r requirements.txt",
            "",
            "## Usage",
            "python main.py"
        ])
    ]
    
    for filename, title, lines in documents:
        img = Image.new('RGB', (500, 400), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw header
        draw.rectangle([0, 0, 500, 40], fill='darkblue')
        draw.text((250, 20), title, fill='white', anchor="mm")
        
        # Draw code/text lines
        y_pos = 60
        for line in lines:
            draw.text((20, y_pos), line, fill='black')
            y_pos += 30
        
        img.save(f"assets/{filename}")
        print(f"   ✅ Created: {filename}")
    
    print(f"\n📸 Successfully created {14} sample images in assets/")
    return True


def create_sample_photos():
    """Create photo-like images for testing"""
    
    photos = [
        ("photo_sunset.png", create_gradient_background(800, 600, (255, 94, 77), (255, 206, 84))),
        ("photo_ocean.png", create_gradient_background(800, 600, (0, 119, 190), (0, 180, 216))),
        ("photo_forest.png", create_gradient_background(800, 600, (34, 139, 34), (107, 142, 35)))
    ]
    
    for filename, img in photos:
        # Add some overlay text
        draw = ImageDraw.Draw(img)
        title = filename.replace("photo_", "").replace(".png", "").title()
        draw.text((400, 300), f"Sample {title} Photo", fill='white', anchor="mm")
        
        img.save(f"assets/{filename}")
        print(f"   ✅ Created: {filename}")
    
    return True


if __name__ == "__main__":
    print("🎨 Creating Sample Assets for Jina Embeddings v4")
    print("=" * 50)
    
    create_sample_images()
    create_sample_photos()
    
    print("\n✅ All sample assets created successfully!")
    print("📁 Check the 'assets/' directory for the generated images")