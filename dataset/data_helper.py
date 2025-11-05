# /workspace/CheXpert/R2GenGPT/dataset/data_helper.py

import os
import json
import re
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from transformers import AutoImageProcessor



class FieldParser:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)
        
        # CheXpert disease labels
        self.disease_labels = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
            'Lung Opacity', 'No Finding', 'Pleural Effusion',
            'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
        ]

    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0]

    def clean_report(self, report):
        """Clean report text"""
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            # FIXED: Use raw string for regex pattern
            sent_cleaner = lambda t: re.sub(r'[.,?;*!%^&_+():\-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        else:
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            # FIXED: Use raw string for regex pattern
            sent_cleaner = lambda t: re.sub(r'[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        return report
    
    def extract_disease_labels(self, report):
        """
        Extract disease labels from report text using keyword matching
        Returns binary vector of shape [14]
        """
        report_lower = report.lower()
        labels = torch.zeros(14, dtype=torch.float32)
        
        # Simple keyword-based extraction
        disease_keywords = {
            0: ['atelectasis', 'collapse'],
            1: ['cardiomegaly', 'enlarged heart', 'cardiac enlargement'],
            2: ['consolidation'],
            3: ['edema', 'pulmonary edema'],
            4: ['enlarged cardiomediastinum', 'mediastinal widening'],
            5: ['fracture', 'rib fracture'],
            6: ['lesion', 'nodule', 'mass'],
            7: ['opacity', 'opacities'],
            8: ['no finding', 'normal', 'clear'],
            9: ['effusion', 'pleural effusion'],
            10: ['pleural'],
            11: ['pneumonia', 'infiltrate'],
            12: ['pneumothorax'],
            13: ['support device', 'catheter', 'tube', 'line']
        }
        
        for idx, keywords in disease_keywords.items():
            if any(kw in report_lower for kw in keywords):
                labels[idx] = 1.0
        
        return labels

    def parse(self, features):
        """Parse features"""
        to_return = {'id': features['id']}
        
        # Parse report
        report = features.get("report", "")
        report = self.clean_report(report)
        to_return['input_text'] = report
        
        # Extract disease labels if enabled
        if self.args.use_disease_labels:
            disease_labels = self.extract_disease_labels(report)
            to_return['disease_labels'] = disease_labels
        
        # Parse images
        images = []
        for image_path in features['image_path']:
            with Image.open(os.path.join(self.args.base_dir, image_path)) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                images.append(image)
        to_return["image"] = images
        
        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        annotation_path = args.annotation
        
        # Load annotation file
        with open(annotation_path, 'r') as f:
            self.meta = json.load(f)
        
        # Handle different annotation formats
        if isinstance(self.meta, dict) and split in self.meta:
            self.meta = self.meta[split]
        elif isinstance(self.meta, list):
            # Filter by split
            self.meta = [item for item in self.meta if item.get('split') == split]
        
        self.parser = FieldParser(args)
        print(f"Loaded {len(self.meta)} samples for {split} split")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'valid')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset
