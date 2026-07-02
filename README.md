# IA Workshop - Modelos de Clasificación: Regresión Logística

## 📚 Parte 1: Investigación y Respuestas Teóricas

Este documento contiene las respuestas detalladas a todas las preguntas sobre algoritmos de clasificación en Machine Learning supervisado.

---

## 1️⃣ ¿Qué es la Clasificación en Machine Learning?

### Definición
La **clasificación** es una tarea de Machine Learning supervisado donde el objetivo es predecir a qué **categoría o clase** pertenece una observación basándose en características conocidas. A diferencia de la regresión (que predice números), la clasificación predice etiquetas discretas.

### Propósito
- Asignar nuevas muestras a categorías predefinidas
- Basarse en patrones aprendidos de datos históricos
- Tomar decisiones automáticas basadas en probabilidades

### Ejemplo Práctico del Mundo Real: Diagnóstico Médico
**Problema:** Predecir si un paciente tiene diabetes o no basándose en sus características médicas.

**Entrada (Características):**
- Nivel de glucosa en sangre
- Presión arterial
- Índice de masa corporal (IMC)
- Edad
- Historial familiar

**Salida (Clase):**
- Clase 0: No tiene diabetes
- Clase 1: Tiene diabetes

**Por qué es importante:** Un sistema automático de clasificación podría ayudar a los médicos a identificar pacientes en riesgo rápidamente, mejorando el diagnóstico temprano.

---

## 2️⃣ Clasificación Binaria vs Multiclase

### Clasificación Binaria

**Definición:** Problemas con exactamente **2 clases mutuamente excluyentes**.

**Características:**
- Salida: 0 o 1 (Falso o Verdadero)
- Más simple de implementar
- Métodos más maduros

**Ejemplos:**
- ✉️ Email: Spam / No Spam
- 🏥 Enfermedad: Positivo / Negativo
- 🎓 Estudiante: Aprobado / Reprobado
- 💳 Transacción: Fraude / Legítima

**Algoritmos típicos:**
- Regresión Logística (la más natural para binaria)
- SVM
- KNN

---

### Clasificación Multiclase

**Definición:** Problemas con **3 o más clases mutuamente excluyentes**.

**Características:**
- Salida: Múltiples opciones (0, 1, 2, 3, ...)
- Más compleja que binaria
- Requiere estrategias especiales

**Ejemplos:**
- 🌺 Iris: Setosa / Versicolor / Virginica
- 🎬 Películas: Comedia / Drama / Acción / Terror / Romántica
- 🍎 Frutas: Manzana / Naranja / Plátano / Pera
- 📮 Correo: Trabajo / Personal / Spam / Promotions

**Algoritmos típicos:**
- Árboles de Decisión
- Random Forest
- Naive Bayes
- SVM (con estrategia one-vs-rest)

---

### Comparación

| Característica | Binaria | Multiclase |
|---|---|---|
| **Número de clases** | 2 | 3+ |
| **Complejidad** | Baja | Media-Alta |
| **Interpretabilidad** | Muy clara | Más compleja |
| **Datos necesarios** | Menos | Más (para cada clase) |
| **Algoritmos** | Específicos para binaria | Genéricos |
| **Cuándo usar** | Sí/No, Aprobado/Reprobado | Múltiples opciones |

---

## 3️⃣ Principales Algoritmos de Clasificación Supervisada

### 📊 Regresión Logística

**¿Qué es?**
A pesar de su nombre, es un algoritmo de **clasificación**, no de regresión. Utiliza una función logística (sigmoide) para transformar una combinación lineal de características en una probabilidad entre 0 y 1.

**Función Matemática:**
```
P(Y=1) = 1 / (1 + e^-(β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ))
```

**Características:**
- ✅ Simple e interpretable
- ✅ Rápido para entrenar
- ✅ Proporciona probabilidades
- ✅ Funciona bien en datos de alta dimensión
- ❌ Asume relaciones lineales
- ❌ No funciona bien con datos no lineales

**Cuándo usar:**
- Problemas binarios simples
- Cuando necesitas interpretabilidad
- Como baseline inicial
- Cuando los datos son linealmente separables

**Ejemplo:** Predecir si un estudiante aprobará basándose en horas de estudio.

---

### 🌳 Árboles de Decisión

**¿Qué es?**
Un algoritmo que crea un árbol de decisiones recursivo, dividiendo el espacio de características en regiones rectangulares basadas en preguntas simples (¿X > 5?).

**Cómo funciona:**
```
¿Horas de estudio > 5?
├─ Sí → ¿Calificación promedio > 7?
│       ├─ Sí → APROBARÁ (probabilidad alta)
│       └─ No → PUEDE REPROBAR
└─ No → PROBABLEMENTE NO APRUEBA
```

**Características:**
- ✅ Muy interpretable (como preguntas "sí/no")
- ✅ No requiere normalización de datos
- ✅ Funciona con datos no lineales
- ✅ Maneja datos categóricos nativamente
- ❌ Propenso a overfitting
- ❌ Inestable (pequeños cambios grandes diferencias)
- ❌ Tiende a ser sesgado con clases desbalanceadas

**Cuándo usar:**
- Cuando necesitas explicabilidad máxima
- Con datos pequeños
- Cuando hay mezcla de variables numéricas y categóricas

---

### 🎯 Support Vector Machine (SVM)

**¿Qué es?**
Un algoritmo que encuentra el **hiperplano óptimo** que maximiza la distancia entre clases. Es como encontrar la línea (o superficie) que mejor separa dos grupos de puntos, dejando el máximo margen posible.

**Visualización:**
```
Clase 0    ──────────────────    Clase 1
  ●●●          MARGEN          ●●●
   ●●              ↕              ●●
     ●     ← Hiperplano →        ●
      ●●●                     ●●●
```

**Características:**
- ✅ Muy poderoso en espacios altos
- ✅ Funciona bien con datos no lineales (kernel trick)
- ✅ Eficiente en memoria
- ✅ Versátil (binaria y multiclase)
- ❌ Complejo de entender
- ❌ Lento con muchos datos
- ❌ Requiere normalización
- ❌ Difícil de interpretar

**Cuándo usar:**
- Con datos de alta dimensión
- Cuando tienes datos limitados
- Problemas complejos no lineales

**Ejemplo:** Reconocimiento de dígitos manuscritos (0-9).

---

### 👥 K-Nearest Neighbors (KNN)

**¿Qué es?**
Un algoritmo perezoso que clasifica una nueva muestra basándose en las **K muestras más cercanas** en el conjunto de entrenamiento. Si 7 de los 10 vecinos más cercanos están en la clase "Aprobado", entonces predice "Aprobado".

**Visualización:**
```
         Nuevo punto ?
              ↓
         Encontrar K=5 vecinos más cercanos
              ↓
      3 son "Aprobado", 2 son "No Aprobado"
              ↓
         Predicción: APROBADO (mayoría)
```

**Características:**
- ✅ Muy simple de entender
- ✅ No asume nada sobre los datos
- ✅ Funciona bien con datos no lineales
- ✅ Funciona con multiclase naturalmente
- ❌ Lento en predicción (debe buscar todos los datos)
- ❌ Sensible a características irrelevantes
- ❌ Sensible a la normalización
- ❌ Necesita mucha memoria

**Cuándo usar:**
- Como baseline rápido
- Con conjuntos de datos pequeños
- Como verificación de sospecha

**Hiperparámetro importante:** K (número de vecinos)
- K pequeño (3-5): Modelo más flexible, más overfitting
- K grande (15-20): Modelo más estable, más underfitting

---

### 🎲 Naive Bayes

**¿Qué es?**
Un algoritmo probabilístico basado en el **Teorema de Bayes**. Asume que todas las características son independientes dado la clase (que raramente es verdad, pero funciona bien en práctica).

**Fórmula de Bayes:**
```
P(Clase|Características) = P(Características|Clase) × P(Clase) / P(Características)
```

**Ejemplo intuitivo:**
```
Dado que horas_estudio=8 y promedio=8:
¿Cuál es la probabilidad de que apruebe?

P(Aprueba|8h, promedio8) = 
    P(8h, promedio8 | Aprueba) × P(Aprueba) / P(8h, promedio8)
```

**Características:**
- ✅ Muy rápido (entrenamiento y predicción)
- ✅ Funciona bien con menos datos
- ✅ Probabilidades interpretables
- ✅ Funciona bien en alta dimensión
- ❌ Asume independencia (generalmente falso)
- ❌ Puede ser sesgado
- ❌ Sensible a características irrelevantes

**Cuándo usar:**
- Clasificación de texto (spam detection, sentiment)
- Datos de alta dimensión con pocas muestras
- Cuando necesitas entrenar rápido

**Variantes:**
- **Gaussian NB:** Para variables continuas (distribuidas normalmente)
- **Multinomial NB:** Para conteos (documentos, palabras)
- **Bernoulli NB:** Para datos binarios

---

## 4️⃣ Métricas para Evaluar Modelos de Clasificación

### 📊 Matriz de Confusión

**Definición:** Tabla que muestra todas las combinaciones de predicciones reales vs predichas.

**Estructura:**
```
                    Predicho Positivo    Predicho Negativo
Realmente Positivo        TP                    FN
Realmente Negativo        FP                    TN

Donde:
- TP (True Positive): Predijo correcto positivo ✓
- TN (True Negative): Predijo correcto negativo ✓
- FP (False Positive): Predijo positivo pero era negativo ✗
- FN (False Negative): Predijo negativo pero era positivo ✗
```

**Ejemplo:** Diagnóstico de enfermedad
```
                    Dice "Enfermo"    Dice "Sano"
Está Enfermo           450 TP             50 FN
Está Sano              30 FP            470 TN

Total: 950 pacientes
```

---

### 🎯 Accuracy (Exactitud)

**Fórmula:**
```
Accuracy = (TP + TN) / Total = (450 + 470) / 950 = 0.971 = 97.1%
```

**Interpretación:** Porcentaje de predicciones correctas en total.

**Cuándo usar:**
- ✅ Cuando las clases están balanceadas
- ✅ Para comparación general rápida
- ❌ No es buena cuando hay desbalance (ej: 1% positivos)

**Problema:** Si un conjunto de datos tiene 99 "No Enfermos" y 1 "Enfermo", un modelo que siempre predice "No Enfermo" tendría 99% accuracy pero sería inútil.

---

### 🎯 Precision (Precisión)

**Fórmula:**
```
Precision = TP / (TP + FP) = 450 / (450 + 30) = 0.937 = 93.7%
```

**Interpretación:** De todos los casos que predije como positivos, ¿cuántos eran realmente positivos?

**Analogía:** Si el modelo dice "tiene enfermedad", ¿en qué porcentaje tiene razón?

**Cuándo usar:**
- ❌ Es muy importante que NO haya falsos positivos
- 📧 Spam detection (no queremos marcar email legítimo como spam)
- 🏥 Alertas médicas (no queremos alarmar innecesariamente)
- 💰 Fraude (no queremos bloquear transacciones legítimas)

---

### 🎯 Recall (Sensibilidad)

**Fórmula:**
```
Recall = TP / (TP + FN) = 450 / (450 + 50) = 0.900 = 90%
```

**Interpretación:** De todos los casos realmente positivos, ¿cuántos detectó mi modelo?

**Analogía:** Si alguien está realmente enfermo, ¿cuál es la probabilidad de que mi modelo lo identifique?

**Cuándo usar:**
- ❌ Es muy importante NO perder positivos
- 🔍 Detección de fraude (no queremos dejar pasar fraudes)
- 🏥 Detección de cáncer (no queremos dejar a nadie sin diagnosticar)
- 🔒 Seguridad (no queremos dejar pasar intrusos)

---

### ⚖️ F1-Score

**Fórmula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.937 × 0.900) / (0.937 + 0.900)
   = 0.918 = 91.8%
```

**Interpretación:** Media armónica entre Precision y Recall. Balance entre ambas.

**Cuándo usar:**
- ✅ Cuando necesitas balance entre Precision y Recall
- ✅ Cuando los datos están desbalanceados
- ✅ Cuando no sabes cuál métrica enfatizar

**Ejemplo:** Un modelo con Precision=100%, Recall=1% tiene F1 muy bajo porque falla en Recall.

---

### 📈 ROC-AUC (Receiver Operating Characteristic - Area Under the Curve)

**¿Qué es?**
Gráfica que muestra el **trade-off entre tasa de verdaderos positivos y falsos positivos** a diferentes umbrales de decisión.

**Visualización:**
```
  TPR (Recall)
   100% ●─────────●  Modelo Perfecto (AUC=1.0)
        │        ╱│
        │      ╱  │
   50%  │    ╱    │  Modelo Bueno (AUC=0.85)
        │  ╱      │
        │╱────────│  Modelo Aleatorio (AUC=0.5)
    0%  ●─────────●
        0%       100% FPR (1-Specificity)
```

**Interpretación AUC:**
- **AUC = 1.0:** Modelo perfecto
- **AUC = 0.85:** Modelo bueno
- **AUC = 0.5:** Modelo aleatorio (no discrimina)
- **AUC < 0.5:** Peor que aleatorio

**Ventaja:** No sensible a cambios de threshold y desbalance de clases.

**Cuándo usar:**
- ✅ Cuando hay desbalance de clases
- ✅ Para comparar modelos en diferentes thresholds
- ✅ Cuando es importante ver el comportamiento general

---

## 5️⃣ Feature Engineering y Feature Selection

### Feature Engineering (Ingeniería de Características)

**Definición:** Crear nuevas características a partir de las existentes para mejorar el modelo.

**Ejemplos:**
```
Características originales:
- Edad: 35 años
- Peso: 75 kg
- Altura: 1.80 m

Feature Engineering:
- IMC = Peso / (Altura²) = 75 / (1.80²) = 23.15 ← NUEVA
- Es_Mayor_Edad = 1 si Edad > 60 else 0 ← NUEVA
- Peso_Normalizado = (Peso - 70) / 10 ← TRANSFORMACIÓN
```

**Técnicas comunes:**
1. **Transformaciones matemáticas:**
   - Logaritmo: log(x) para datos sesgados
   - Raíz cuadrada: √x
   - Polinomios: x², x³

2. **Agrupación:**
   - Categorizar edad en grupos: Joven (18-30), Adulto (30-50), Mayor (50+)

3. **Interacciones:**
   - Multiplicar características: edad × ingresos
   - Dividir: ingresos / gastos

4. **Extracción temporal:**
   - Día de la semana, mes, año de una fecha
   - Hora del día, fin de semana o laboral

**Importancia:** Las características que usas determinan el techo de desempeño.

---

### Feature Selection (Selección de Características)

**Definición:** Elegir un **subconjunto de características** relevantes y descartar las irrelevantes.

**Por qué es importante:**
- ✅ Reduce overfitting
- ✅ Mejora interpretabilidad
- ✅ Reduce tiempo de entrenamiento
- ✅ Mejora generalización

**Métodos:**

1. **Métodos Univariados (cada variable por su cuenta):**
   - Correlación con la clase objetivo
   - Chi-cuadrado (para categóricas)
   - ANOVA (para numéricas)

2. **Métodos de Eliminación:**
   - Eliminar la característica menos importante iterativamente
   - Backwards elimination

3. **Métodos de Selección:**
   - Agregar características una a una
   - Forward selection

4. **Basados en Modelos:**
   - Usar importancia del modelo (Random Forest)
   - Coeficientes de regresión

**Ejemplo:**
```
Conjunto inicial: 20 características
↓ Análisis de correlación
Descubrimiento: 5 características tienen correlación < 0.1 con objetivo
↓ Eliminación
Conjunto final: 15 características más relevantes
↓ Resultado: Mejor desempeño, modelo más simple
```

---

## 7️⃣ Hiperparámetros y Optimización

### ¿Qué son los Hiperparámetros?

**Definición:** Parámetros que se configuran ANTES de entrenar el modelo. No se aprenden, se ajustan manualmente.

**Diferencia con parámetros:**
- **Parámetros:** Se aprenden durante el entrenamiento (ej: coeficientes β en regresión)
- **Hiperparámetros:** Se configuran antes (ej: tasa de aprendizaje, número de iteraciones)

**Ejemplos por algoritmo:**

| Algoritmo | Hiperparámetros |
|---|---|
| **Regresión Logística** | C (regularización), solver, max_iter |
| **Árbol de Decisión** | max_depth, min_samples_split, min_samples_leaf |
| **SVM** | C, kernel (linear, rbf), gamma |
| **KNN** | n_neighbors (K), metric (distancia) |
| **Naive Bayes** | alpha (suavizado) |

---

### Grid Search (Búsqueda en Malla)

**¿Qué es?** Prueba **todas las combinaciones posibles** de hiperparámetros.

**Ejemplo:**
```python
Parámetros a probar:
- C: [0.1, 1, 10]
- max_iter: [100, 200, 300]

Combinaciones totales: 3 × 3 = 9 modelos

Busca exhaustiva:
1. C=0.1, max_iter=100 → Accuracy = 0.85
2. C=0.1, max_iter=200 → Accuracy = 0.86
3. C=0.1, max_iter=300 → Accuracy = 0.86
4. C=1, max_iter=100   → Accuracy = 0.87
... (total 9)

Mejor: C=10, max_iter=300 → Accuracy = 0.92
```

**Ventajas:**
- ✅ Garantiza encontrar la mejor combinación
- ✅ Simple de entender
- ✅ Paralelizable

**Desventajas:**
- ❌ Muy lento con muchos parámetros (explosión combinatoria)
- ❌ Puede ser computacionalmente costoso

**Cuándo usar:**
- Con pocos hiperparámetros (2-3)
- Con espacio de búsqueda pequeño
- Cuando tienes recursos computacionales

---

### Random Search (Búsqueda Aleatoria)

**¿Qué es?** Prueba **combinaciones aleatorias** de hiperparámetros.

**Ejemplo:**
```python
Parámetros a probar:
- C: [0.01, 0.1, 1, 10, 100]
- max_iter: [50, 100, 200, 300, 500]

Iteraciones: 20 pruebas aleatorias

Busca aleatoria:
1. C=0.1, max_iter=200 → Accuracy = 0.86
2. C=10, max_iter=300  → Accuracy = 0.92
3. C=1, max_iter=100   → Accuracy = 0.87
... (total 20 pruebas de 25 posibles)

Mejor encontrado: C=10, max_iter=300 → Accuracy = 0.92
```

**Ventajas:**
- ✅ Más eficiente que Grid Search en espacios grandes
- ✅ Escalable a más parámetros
- ✅ Menos computacionalmente costoso

**Desventajas:**
- ❌ No garantiza encontrar el óptimo
- ❌ Menos determinístico

**Cuándo usar:**
- Con muchos hiperparámetros
- Cuando tienes espacio de búsqueda grande
- Como exploración inicial antes de Grid Search

---

### Comparación

| Aspecto | Grid Search | Random Search |
|---|---|---|
| **Cobertura** | 100% | Parcial |
| **Velocidad** | Lenta (exponencial) | Rápida (lineal) |
| **Garantía** | Encuentra óptimo | No garantiza |
| **Parámetros** | 2-3 ideales | 4+ óptimo |
| **Tiempo tuning** | Largo | Corto |

**Estrategia híbrida:** Usar Random Search primero para explorar, luego Grid Search en la región prometedora.

---

## 📌 Resumen General de Algoritmos

| Algoritmo | Velocidad | Interpretable | Lineal | Datos | Cuándo |
|---|---|---|---|---|---|
| **Regresión Logística** | ⭐⭐⭐ | ⭐⭐⭐ | ✓ | Pocos | Baseline, binaria simple |
| **Árbol Decisión** | ⭐⭐ | ⭐⭐⭐ | ✗ | Pocos | Explicabilidad máxima |
| **SVM** | ⭐⭐ | ⭐ | ✗ | Muchos | Datos alta dimensión |
| **KNN** | ⭐ | ⭐⭐ | ✗ | Pocos | Baseline rápido |
| **Naive Bayes** | ⭐⭐⭐ | ⭐⭐ | - | Muchos | Text, alta dimensión |

---

## 📚 Conclusión

Este documento proporciona la base teórica necesaria para entender la clasificación en Machine Learning. Ahora, con el notebook práctico, puedes experimentar cómo estos conceptos funcionan en la práctica.

**Próximos pasos después de este workshop:**
1. Trabajar con datasets reales (Iris, Titanic, etc.)
2. Explorar técnicas de preprocesamiento
3. Aprender validación cruzada y ajuste de hiperparámetros
4. Comparar múltiples algoritmos
5. Trabajar en proyectos integrales

---

## 🔗 Referencias

- Scikit-learn Documentation: https://scikit-learn.org/
- Machine Learning Mastery: https://machinelearningmastery.com/
- Google Machine Learning Crash Course: https://developers.google.com/machine-learning/crash-course

---

**Autor:** Workshop de IA - Factoria F5  
**Fecha:** 2024  
**Nivel:** Principiante  
**Requisitos:** Python, NumPy, Scikit-learn
