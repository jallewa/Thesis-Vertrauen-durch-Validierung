# Vertrauen durch Validierung: XAI-Evaluation in der Dermatologie

Dieses Repository enthält den offiziellen Quellcode und das zugehörige Jupyter Notebook zur Bachelorarbeit: **"Vertrauen durch Validierung: Eine quantitative Evaluation der Robustheit und Erklärungstreue von XAI-Methoden bei der dermatologischen Bildklassifikation auf dem HAM10000-Datensatz"**.

## Über das Projekt
Der Einsatz von Deep-Learning-Modellen, insbesondere Convolutional Neural Networks (CNNs), zeigt bei der Hautkrebsdiagnostik beeindruckende Ergebnisse. Da diese Modelle jedoch als "Black-Box" agieren, mangelt es in der klinischen Praxis oft an Vertrauen. Explainable AI (XAI) Methoden sollen diese Modelle transparent machen, indem sie Erklärungen in Form von Heatmaps liefern.

Dieses Projekt evaluiert systematisch und *quantitativ*, wie verlässlich (erklärungstreu) und robust diese XAI-Methoden wirklich sind [4]. Um die Abhängigkeit von der Modellarchitektur zu untersuchen, werden ein massives, tiefes Netzwerk (**ResNet-101**) und ein leichtgewichtiges, für mobile Endgeräte optimiertes Netzwerk (**MobileNetV3**) gegenübergestellt.

## Methodik & Pipeline
Die experimentelle Pipeline in diesem Notebook durchläuft folgende Schritte [7]:

1. **Datensatz & Preprocessing:** Verwendung des **HAM10000-Datensatzes**, der 10.015 dermatoskopische Bilder der sieben häufigsten pigmentierten Hautläsionen umfasst. Die Klassenimbalance wird während des Trainings durch Oversampling-Techniken behoben. Für das Netzwerk werden die Bilder entsprechend der ImageNet-Verteilung normalisiert.
2. **Modell-Training (Transfer Learning):** Feinabstimmung (Fine-Tuning) von ResNet-101 und MobileNetV3 (vorab trainiert auf ImageNet) zur Klassifikation der Läsionen. Das finale Modell erreicht auf einem strikt separierten Testdatensatz (1503 Bilder, 15 % Split) eine Accuracy von über 86 %.
3. **Erklärungsgenerierung (XAI):** Öffnen der "Black-Box" durch die Anwendung zweier konzeptionell unterschiedlicher XAI-Methoden auf ausgewählte Testbilder:
   * **Grad-CAM** (Gradienten-basiert)
   * **LIME** (Perturbations-basiert)
4. **Quantitative Evaluation:** Nutzung des **Quantus-Frameworks** zur standardisierten Black-Box-Evaluierung:
   * **Erklärungstreue (Faithfulness):** Gemessen mit der **IROF**-Metrik (Iterative Removal of Features), um das Pixel-Korrelations-Problem hochauflösender medizinischer Bilder zu lösen [13].
   * **Robustheit (Robustness):** Gemessen über **Local Lipschitz Estimate** / **Relative Output Stability**, um die Stabilität der Erklärungen bei minimalen, künstlichen Bildstörungen (Rauschen) zu quantifizieren [14, 15].

### Autor
Christopher Böhm
Wilhelm Büchner Hochschule (WBH)

Studiengang: Informatik (B.Sc.)

Kontakt: christopher.boehm@student.wb-hochschule.com