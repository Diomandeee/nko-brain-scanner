/**
 * Batch Job Configuration for N'Ko Training Data Extraction
 * 
 * This script creates and submits batch jobs for processing N'Ko educational videos.
 * It uses the deployed Cloud Run backend's batch API.
 */

import fs from 'fs';
import path from 'path';

// Backend URL
const BACKEND_URL = process.env.ANALYZER_BACKEND_URL || 'https://cc-music-pipeline-owq2vk3wya-uc.a.run.app';

// Batch configuration
export const BATCH_CONFIG = {
  // Processing settings
  frameRate: 0.2,                    // 1 frame every 5 seconds (matches video slide transitions)
  deduplicationThreshold: 0.85,      // 85% similarity = duplicate
  maxFramesPerVideo: 100,            // Cap frames per video
  
  // Output settings
  outputFormat: 'jsonl' as const,
  template: 'nko_ocr' as const,
  
  // Batching settings
  videosPerBatch: 20,                // 20 videos per batch job
  maxConcurrentBatches: 3,           // Don't overload the API
  
  // Cost estimate (based on Gemini pricing)
  estimatedFramesPerVideo: 60,       // Average video ~5 min = 60 frames at 0.2fps
  pricePerFrame: 0.00065,            // ~$0.65 per 1000 images with batch discount
};

// Load video manifest
export function loadVideoManifest(): VideoManifest {
  const manifestPath = path.join(__dirname, '../data/video_manifest.json');
  const data = fs.readFileSync(manifestPath, 'utf-8');
  return JSON.parse(data);
}

// Types
export interface VideoEntry {
  id: string;
  url: string;
  title: string;
}

export interface VideoManifest {
  channel: string;
  total_videos: number;
  scraped_at: string;
  videos: VideoEntry[];
}

export interface BatchJobRequest {
  mode: 'training_data_generation';
  video_urls: string[];
  frame_rate: number;
  deduplication_threshold: number;
  format: 'jsonl';
  priority: 'standard' | 'high';
}

export interface BatchJobResponse {
  success: boolean;
  job_id: string;
  job_name: string;
  status: string;
  request_count: number;
  estimated_cost_usd: number;
  submitted_at: string;
}

// Create batch requests from video manifest
export function createBatchRequests(manifest: VideoManifest): BatchJobRequest[] {
  const batches: BatchJobRequest[] = [];
  const { videos } = manifest;
  const { videosPerBatch } = BATCH_CONFIG;
  
  for (let i = 0; i < videos.length; i += videosPerBatch) {
    const chunk = videos.slice(i, i + videosPerBatch);
    batches.push({
      mode: 'training_data_generation',
      video_urls: chunk.map(v => v.url),
      frame_rate: BATCH_CONFIG.frameRate,
      deduplication_threshold: BATCH_CONFIG.deduplicationThreshold,
      format: BATCH_CONFIG.outputFormat,
      priority: 'standard',
    });
  }
  
  return batches;
}

// Submit a single batch job
export async function submitBatchJob(request: BatchJobRequest): Promise<BatchJobResponse> {
  const response = await fetch(`${BACKEND_URL}/api/batch/training/submit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to submit batch job: ${error}`);
  }
  
  return response.json();
}

// Check batch job status
export async function checkJobStatus(jobId: string): Promise<{ status: string; progress?: number }> {
  const response = await fetch(`${BACKEND_URL}/api/batch/status/${jobId}`);
  
  if (!response.ok) {
    throw new Error(`Failed to check job status: ${response.statusText}`);
  }
  
  return response.json();
}

// Get batch job results
export async function getJobResults(jobId: string): Promise<any> {
  const response = await fetch(`${BACKEND_URL}/api/batch/results/${jobId}`);
  
  if (!response.ok) {
    throw new Error(`Failed to get job results: ${response.statusText}`);
  }
  
  return response.json();
}

// Cost estimation
export function estimateCost(videoCount: number): {
  estimatedFrames: number;
  estimatedCost: number;
  batchDiscount: number;
  finalCost: number;
} {
  const estimatedFrames = videoCount * BATCH_CONFIG.estimatedFramesPerVideo;
  const estimatedCost = estimatedFrames * BATCH_CONFIG.pricePerFrame;
  const batchDiscount = estimatedCost * 0.5; // 50% batch discount
  const finalCost = estimatedCost - batchDiscount;
  
  return {
    estimatedFrames,
    estimatedCost,
    batchDiscount,
    finalCost,
  };
}

// Main execution (for CLI usage)
async function main() {
  const args = process.argv.slice(2);
  const command = args[0];
  
  console.log('='.repeat(60));
  console.log('N\'Ko Training Data Batch Processor');
  console.log('='.repeat(60));
  
  // Load manifest
  const manifest = loadVideoManifest();
  console.log(`\nLoaded ${manifest.total_videos} videos from ${manifest.channel}`);
  
  // Calculate cost
  const cost = estimateCost(manifest.total_videos);
  console.log('\nCost Estimate:');
  console.log(`  Estimated frames: ${cost.estimatedFrames.toLocaleString()}`);
  console.log(`  Base cost: $${cost.estimatedCost.toFixed(2)}`);
  console.log(`  Batch discount (50%): -$${cost.batchDiscount.toFixed(2)}`);
  console.log(`  Final cost: $${cost.finalCost.toFixed(2)}`);
  
  // Create batch requests
  const batches = createBatchRequests(manifest);
  console.log(`\nBatch Plan:`);
  console.log(`  Total batches: ${batches.length}`);
  console.log(`  Videos per batch: ${BATCH_CONFIG.videosPerBatch}`);
  console.log(`  Frame rate: ${BATCH_CONFIG.frameRate} fps`);
  
  if (command === 'test') {
    // Submit only first batch as test
    console.log('\n📤 Submitting TEST batch (first 10 videos)...');
    const testBatch = {
      ...batches[0],
      video_urls: batches[0].video_urls.slice(0, 10),
    };
    
    try {
      const result = await submitBatchJob(testBatch);
      console.log(`\n✅ Test batch submitted successfully!`);
      console.log(`   Job ID: ${result.job_id}`);
      console.log(`   Estimated cost: $${result.estimated_cost_usd.toFixed(2)}`);
      console.log(`   Request count: ${result.request_count}`);
    } catch (error) {
      console.error(`\n❌ Failed to submit test batch:`, error);
    }
  } else if (command === 'submit-all') {
    console.log('\n📤 Submitting ALL batches...');
    console.log('⚠️  This will process all 532 videos!');
    console.log('Press Ctrl+C within 5 seconds to cancel...\n');
    
    await new Promise(r => setTimeout(r, 5000));
    
    for (let i = 0; i < batches.length; i++) {
      console.log(`\nBatch ${i + 1}/${batches.length}:`);
      try {
        const result = await submitBatchJob(batches[i]);
        console.log(`  ✅ Job ${result.job_id} submitted`);
        
        // Wait between batches
        if (i < batches.length - 1) {
          console.log('  ⏳ Waiting 30s before next batch...');
          await new Promise(r => setTimeout(r, 30000));
        }
      } catch (error) {
        console.error(`  ❌ Failed:`, error);
      }
    }
  } else {
    console.log('\nUsage:');
    console.log('  npx ts-node batch-config.ts test        - Submit test batch (10 videos)');
    console.log('  npx ts-node batch-config.ts submit-all  - Submit all batches');
  }
}

// Export for module usage
export default {
  BATCH_CONFIG,
  loadVideoManifest,
  createBatchRequests,
  submitBatchJob,
  checkJobStatus,
  getJobResults,
  estimateCost,
};

// Run if executed directly
if (require.main === module) {
  main().catch(console.error);
}

