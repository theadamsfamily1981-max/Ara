/**
 * @file emit_emotion.h
 * @brief Emotion emission protocol for FPGA/HPS → Host communication
 *
 * Iteration 46: First Hardware-Ready Organism Harness
 *
 * This header provides functions to emit emotional state, HPV classifications,
 * and memory events from the FPGA/HPS side to the host emotion_bridge daemon.
 *
 * Wire Protocol:
 *   EMO {"emotion":"...", "strength":0.0, "valence":0.0, "arousal":0.0,
 *        "dominance":0.0, "sparsity":0.0, "homeo_dev":0.0, "tags":["..."]}
 *
 *   HPV {"id":0, "anomaly_score":0.0, "class":"...", "tag":"..."}
 *
 *   EMO_STORE {"index":0, "emotion":"...", "strength":0.0}
 *   EMO_RECALL {"index":0, "emotion":"...", "sim":0.0, "strength":0.0}
 *   EMO_DREAM {"index":0, "sim":0.0, "strength":0.0}
 *
 * Output goes to stdout (wire to UART or named pipe as needed).
 */

#ifndef EMIT_EMOTION_H
#define EMIT_EMOTION_H

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of tags to emit */
#define MAX_TAGS 8

/* Maximum tag string length */
#define MAX_TAG_LEN 32

/* ============================================================================
 * Emotion State Emission
 * ============================================================================ */

/**
 * @brief Emit a full emotional state update
 *
 * @param emotion_name  Name of the emotion (e.g., "RAGE", "calm", "fear")
 * @param strength      Emotion intensity [0.0, 1.0]
 * @param valence       VAD valence [-1.0, +1.0]
 * @param arousal       VAD arousal [-1.0, +1.0]
 * @param dominance     VAD dominance [-1.0, +1.0]
 * @param sparsity      Network sparsity ratio [0.0, 1.0]
 * @param homeo_dev     Homeostasis deviation [0.0, 1.0]
 * @param tags          Array of tag strings (NULL-terminated or use num_tags)
 * @param num_tags      Number of tags to emit
 */
static inline void emit_emotion_state(
    const char* emotion_name,
    float strength,
    float valence,
    float arousal,
    float dominance,
    float sparsity,
    float homeo_dev,
    const char** tags,
    int num_tags
) {
    printf("EMO {\"emotion\":\"%s\",", emotion_name);
    printf("\"strength\":%.3f,", strength);
    printf("\"valence\":%.3f,", valence);
    printf("\"arousal\":%.3f,", arousal);
    printf("\"dominance\":%.3f,", dominance);
    printf("\"sparsity\":%.3f,", sparsity);
    printf("\"homeo_dev\":%.3f,", homeo_dev);
    printf("\"tags\":[");

    for (int i = 0; i < num_tags && i < MAX_TAGS; i++) {
        if (tags[i] != NULL) {
            printf("\"%s\"", tags[i]);
            if (i + 1 < num_tags && tags[i + 1] != NULL) {
                printf(",");
            }
        }
    }

    printf("]}\n");
    fflush(stdout);
}

/**
 * @brief Emit a simple emotional state (no tags)
 */
static inline void emit_emotion_simple(
    const char* emotion_name,
    float strength,
    float valence,
    float arousal,
    float dominance
) {
    emit_emotion_state(
        emotion_name,
        strength,
        valence,
        arousal,
        dominance,
        0.0f,  /* sparsity */
        0.0f,  /* homeo_dev */
        NULL,  /* tags */
        0      /* num_tags */
    );
}

/* ============================================================================
 * HPV Classification Emission
 * ============================================================================ */

/**
 * @brief Emit an HPV classification event
 *
 * @param hpv_id        Unique ID for this HPV
 * @param anomaly_score Anomaly score [0.0, 1.0]
 * @param class_name    Classification ("NORMAL" or "ANOMALY")
 * @param tag           Associated tag/concept name
 */
static inline void emit_hpv_classification(
    int32_t hpv_id,
    float anomaly_score,
    const char* class_name,
    const char* tag
) {
    printf("HPV {\"id\":%d,", hpv_id);
    printf("\"anomaly_score\":%.3f,", anomaly_score);
    printf("\"class\":\"%s\",", class_name);
    printf("\"tag\":\"%s\"}\n", tag ? tag : "");
    fflush(stdout);
}

/* ============================================================================
 * Memory Event Emission
 * ============================================================================ */

/**
 * @brief Emit a memory store event
 *
 * @param index         Memory index where stored
 * @param emotion       Emotion name associated with memory
 * @param strength      Emotional strength of the memory
 */
static inline void emit_memory_store(
    int32_t index,
    const char* emotion,
    float strength
) {
    printf("EMO_STORE {\"index\":%d,", index);
    printf("\"emotion\":\"%s\",", emotion ? emotion : "");
    printf("\"strength\":%.3f}\n", strength);
    fflush(stdout);
}

/**
 * @brief Emit a memory recall event
 *
 * @param index         Memory index that was recalled
 * @param emotion       Emotion associated with recalled memory
 * @param similarity    Cosine similarity to current state
 * @param strength      Stored emotional strength
 */
static inline void emit_memory_recall(
    int32_t index,
    const char* emotion,
    float similarity,
    float strength
) {
    printf("EMO_RECALL {\"index\":%d,", index);
    printf("\"emotion\":\"%s\",", emotion ? emotion : "");
    printf("\"sim\":%.3f,", similarity);
    printf("\"strength\":%.3f}\n", strength);
    fflush(stdout);
}

/**
 * @brief Emit a dream replay event
 *
 * @param index         Memory index being replayed
 * @param similarity    Self-recall similarity
 * @param strength      Stored emotional strength
 */
static inline void emit_memory_dream(
    int32_t index,
    float similarity,
    float strength
) {
    printf("EMO_DREAM {\"index\":%d,", index);
    printf("\"sim\":%.3f,", similarity);
    printf("\"strength\":%.3f}\n", strength);
    fflush(stdout);
}

/* ============================================================================
 * Convenience Macros
 * ============================================================================ */

/**
 * @brief Quick emotion emit with common emotions
 */
#define EMIT_CALM(strength, dominance) \
    emit_emotion_simple("calm", (strength), 0.5f, -0.3f, (dominance))

#define EMIT_ALERT(strength, arousal) \
    emit_emotion_simple("vigilance", (strength), 0.1f, (arousal), 0.5f)

#define EMIT_FEAR(strength, arousal) \
    emit_emotion_simple("fear", (strength), -0.7f, (arousal), -0.6f)

#define EMIT_RAGE(strength, dominance) \
    emit_emotion_simple("rage", (strength), -0.8f, 0.9f, (dominance))

#define EMIT_JOY(strength) \
    emit_emotion_simple("joy", (strength), 0.8f, 0.5f, 0.5f)

/* ============================================================================
 * Example Usage
 * ============================================================================ */

#ifdef EMIT_EMOTION_EXAMPLE
/*
 * Example usage in your HPS/FPGA code:
 *
 * #include "emit_emotion.h"
 *
 * void update_emotional_state(float hidden_rate, float homeo_dev, ...) {
 *     // Compute VAD from fabric physiology
 *     float valence = compute_valence(confidence, stress);
 *     float arousal = compute_arousal(hidden_rate, inhibition);
 *     float dominance = compute_dominance(homeo_dev, sparsity);
 *
 *     // Determine emotion archetype
 *     const char* emotion = classify_emotion(valence, arousal, dominance);
 *     float strength = compute_strength(valence, arousal, dominance);
 *
 *     // Emit to host
 *     const char* tags[] = {"route_flap", "stressed"};
 *     emit_emotion_state(
 *         emotion,
 *         strength,
 *         valence,
 *         arousal,
 *         dominance,
 *         sparsity,
 *         homeo_dev,
 *         tags,
 *         2
 *     );
 * }
 *
 * void on_memory_store(int idx, const char* emo, float str) {
 *     emit_memory_store(idx, emo, str);
 * }
 *
 * void on_memory_recall(int idx, const char* emo, float sim, float str) {
 *     emit_memory_recall(idx, emo, sim, str);
 * }
 */
#endif /* EMIT_EMOTION_EXAMPLE */

#ifdef __cplusplus
}
#endif

#endif /* EMIT_EMOTION_H */
