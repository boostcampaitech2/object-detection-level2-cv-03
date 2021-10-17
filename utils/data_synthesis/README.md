
### 기존 train.json file train_json으로 load

```python3

with open('dataset/train.json') as json_file:
    train_json = json.load(json_file)

```

# Usage
1. bbox_cut.py

class setting ex)
```python3
classes = {"General_trash":0, "Paper":1, "Paper_pack":2, "Metal":3, "Glass":4, 
           "Plastic":5, "Styrofoam":6, "Plastic_bag":7, "Battery":8, "Clothing":9}

classes_invert = {0:"General_trash", 1:"Paper", 2:"Paper_pack", 3:"Metal", 4:"Glass", 
           5:"Plastic", 6:"Styrofoam", 7:"Plastic_bag", 8:"Battery", 9:"Clothing"}
```
모델 config의 class에 맞게 수정 필요
<br>
<br>

2. collage.py

White canvas, Background pathch canvas 2가지 버전 있음.(BG patch data는 해당 레포에서 제공X)

```python3
    #white canvas
    im = Image.new('RGB', (1024, 1024), 'white')
    im.format = "JPG"
    
    #background canvas
    bg_path = 'dataset/BG/'+ bg_list[j]
    im = Image.open(bg_path)
    im.format = "JPG"
``` 

### class 조합 생성

```python3
comb = combinations(["General_trash", "Paper", "Paper_pack", "Metal","Paper", "Paper_pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic_bag", "Clothing","Battery","Battery","Battery"], 4)
``` 
