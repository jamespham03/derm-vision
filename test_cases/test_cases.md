# DermVision — Test Cases

10 sample cases from the ISIC 2019 dataset covering all 8 classes.
Images are in the `images/` folder alongside this document.

> **Note:** The questionnaire answers populate the Patient Profile display only — the AI prediction is driven entirely by the image.

---

## Case 1 — Melanocytic Nevus (NV) · LOW RISK
**Image:** `images/ISIC_0000000.jpg`

| Field | Answer |
|---|---|
| Age | 55 |
| Biological Sex | Female |
| Fitzpatrick Skin Type | Unknown / Not sure |
| Lesion Location | Chest / Upper Back |
| Duration | Unknown |
| Changed recently? | Unsure |
| Symptoms | No symptoms |
| Family history of skin cancer? | Unknown |

**Expected diagnosis:** NV — Melanocytic Nevus (LOW RISK)

---

## Case 2 — Melanoma (MEL) · HIGH RISK
**Image:** `images/ISIC_0000002.jpg`

| Field | Answer |
|---|---|
| Age | 60 |
| Biological Sex | Female |
| Fitzpatrick Skin Type | Unknown / Not sure |
| Lesion Location | Forearm / Elbow |
| Duration | Unknown |
| Changed recently? | Unsure |
| Symptoms | No symptoms |
| Family history of skin cancer? | Unknown |

**Expected diagnosis:** MEL — Melanoma (HIGH RISK)

---

## Case 3 — Benign Keratosis (BKL) · LOW RISK
**Image:** `images/ISIC_0010491.jpg`

| Field | Answer |
|---|---|
| Age | 75 |
| Biological Sex | Female |
| Fitzpatrick Skin Type | Unknown / Not sure |
| Lesion Location | Head / Neck / Face |
| Duration | > 3 years |
| Changed recently? | No |
| Symptoms | No symptoms |
| Family history of skin cancer? | Unknown |

**Expected diagnosis:** BKL — Benign Keratosis (LOW RISK)

---

## Case 4 — Dermatofibroma (DF) · LOW RISK
**Image:** `images/ISIC_0024318.jpg`

| Field | Answer |
|---|---|
| Age | 65 |
| Biological Sex | Female |
| Fitzpatrick Skin Type | Unknown / Not sure |
| Lesion Location | Lower Leg / Knee |
| Duration | Unknown |
| Changed recently? | Unsure |
| Symptoms | No symptoms |
| Family history of skin cancer? | Unknown |

**Expected diagnosis:** DF — Dermatofibroma (LOW RISK)

---

## Case 5 — Squamous Cell Carcinoma (SCC) · HIGH RISK
**Image:** `images/ISIC_0024329.jpg`

| Field | Answer |
|---|---|
| Age | 75 |
| Biological Sex | Female |
| Fitzpatrick Skin Type | Unknown / Not sure |
| Lesion Location | Lower Leg / Knee |
| Duration | 1–6 months |
| Changed recently? | Yes |
| Symptoms | Itching, Color change |
| Family history of skin cancer? | Unknown |

**Expected diagnosis:** SCC — Squamous Cell Carcinoma (HIGH RISK)

---

## Case 6 — Basal Cell Carcinoma (BCC) · HIGH RISK
**Image:** `images/ISIC_0024331.jpg`

| Field | Answer |
|---|---|
| Age | 65 |
| Biological Sex | Male |
| Fitzpatrick Skin Type | Unknown / Not sure |
| Lesion Location | Lower Leg / Knee |
| Duration | 1–3 years |
| Changed recently? | Unsure |
| Symptoms | No symptoms |
| Family history of skin cancer? | Unknown |

**Expected diagnosis:** BCC — Basal Cell Carcinoma (HIGH RISK)

---

## Case 7 — Vascular Lesion (VASC) · LOW RISK
**Image:** `images/ISIC_0024370.jpg`

| Field | Answer |
|---|---|
| Age | 55 |
| Biological Sex | Male |
| Fitzpatrick Skin Type | Unknown / Not sure |
| Lesion Location | Other / Unknown |
| Duration | Unknown |
| Changed recently? | Unsure |
| Symptoms | No symptoms |
| Family history of skin cancer? | Unknown |

**Expected diagnosis:** VASC — Vascular Lesion (LOW RISK)

---

## Case 8 — Actinic Keratosis / Bowen's (AKIEC) · MODERATE RISK
**Image:** `images/ISIC_0024468.jpg`

| Field | Answer |
|---|---|
| Age | 75 |
| Biological Sex | Male |
| Fitzpatrick Skin Type | Type II — Usually burns, sometimes tans |
| Lesion Location | Head / Neck / Face |
| Duration | > 3 years |
| Changed recently? | Yes |
| Symptoms | Itching, Color change |
| Family history of skin cancer? | Unknown |

**Expected diagnosis:** AKIEC — Actinic Keratosis / Bowen's (MODERATE RISK)

---

## Case 9 — Melanocytic Nevus, young female (NV) · LOW RISK
**Image:** `images/ISIC_0000001.jpg`

| Field | Answer |
|---|---|
| Age | 30 |
| Biological Sex | Female |
| Fitzpatrick Skin Type | Unknown / Not sure |
| Lesion Location | Chest / Upper Back |
| Duration | Unknown |
| Changed recently? | No |
| Symptoms | No symptoms |
| Family history of skin cancer? | No |

**Expected diagnosis:** NV — Melanocytic Nevus (LOW RISK)

---

## Case 10 — Melanocytic Nevus, young male (NV) · LOW RISK
**Image:** `images/ISIC_0000003.jpg`

| Field | Answer |
|---|---|
| Age | 30 |
| Biological Sex | Male |
| Fitzpatrick Skin Type | Unknown / Not sure |
| Lesion Location | Forearm / Elbow |
| Duration | Unknown |
| Changed recently? | No |
| Symptoms | No symptoms |
| Family history of skin cancer? | No |

**Expected diagnosis:** NV — Melanocytic Nevus (LOW RISK)
