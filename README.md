# Harris Corner Detection

Este projeto implementa e demonstra de forma **didática e passo a passo** o algoritmo de **Harris Corner Detection**. O foco principal é **facilitar o entendimento do algoritmo**, apresentando visualizações intermediárias, explicações teóricas e análises experimentais.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

# 📘 Objetivos do Trabalho

* Explicar o **processo completo** do algoritmo de Harris, desde o pré-processamento até a detecção final de cantos.
* Separar claramente:

  * O **código principal** da técnica (implementado em Python);
  * Um **notebook didático** para uso em sala de aula;
  * Um **script de experimentos** que explora diferentes parâmetros.
* Produzir visualizações claras que evidenciem:

  * Gradientes,
  * Componentes da matriz de estrutura,
  * Resposta de Harris,
  * Cantos detectados.

---

# 📁 Estrutura do Projeto

```
harris_project/
│
├── harris_core.py               # Implementação completa do algoritmo de Harris
├── harris_demo.ipynb            # Notebook didático (passo a passo)
├── run_harris_experiments.py    # Script para experimentos e resultados finais
│
├── images/
│   └── ...                      # Imagem de teste utilizada
│
└── outputs/
    └── ...                      # Figuras geradas pelos experimentos
```

---

# ⚙️ INSTALAÇÃO

## 📦 Requisitos

* Python **3.9 ou superior**
* Pip atualizado
* Ambiente virtual recomendado (`python -m venv .venv`)

---

## 📥 Instalar dependências

Após ativar o ambiente virtual:

```bash
pip install -r requirements.txt
```

O arquivo `requirements.txt` contém apenas as bibliotecas utilizadas no projeto, incluindo:

```
opencv-python
numpy
matplotlib
jupyter
```

Caso utilize widgets interativos no notebook:

```
ipywidgets
```

---

# ▶️ EXECUÇÃO

## 1. Notebook didático (recomendado)

Execute:

```
harris_demo.ipynb
```

No VSCode, Jupyter Notebook ou Jupyter Lab.

O notebook:

* Carrega o pipeline completo do algoritmo,
* Explica cada etapa em Markdown,
* Exibe gradientes, matriz de estrutura, resposta R,
* Mostra a sobreposição final dos cantos detectados.

---

## 2. Executar experimentos (figuras comparativas)

No terminal:

```bash
python run_harris_experiments.py
```

Resultados serão salvos automaticamente em:

```
outputs/
```

Esse script realiza testes variando:

* Limiar (`threshold_ratio`)
* Parâmetro `k`
* Sigma da suavização

Ideal para análise de sensibilidade e visualização de comportamento.

## 🎬 Vídeo de Demonstração

[https://youtu.be/L0topMASb7c](Harris Corner Detection – Visualização Didática Passo a Passo)

---

# 📚 Conceitos Abordados

O projeto demonstra passo a passo:

* Derivadas de intensidade (I_x, I_y)
* Matriz de autocorrelação (structure tensor)
* Autovalores e interpretação de bordas, cantos e regiões planas
* Função de resposta de Harris
* Limiarização e non-maximum suppression

Essas etapas são fundamentais para entender detectores clássicos de pontos de interesse em visão computacional.

---

# ✔️ Autor

**Wérikson Frederiko de Oliveira Alves**
Disciplina INF792 — Pós-Graduação em Ciência da Computação
Universidade Federal de Viçosa (UFV)

---
