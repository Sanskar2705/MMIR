import sys
import os

# Add project root to Python path
sys.path.append('/mnt/storage/RSystemsBenchmarking/gitProject')

from retrieval_orchestrator import TwoStageRetrievalOrchestrator

def get_user_choice(prompt, options, default=None):
    """Helper function to get user choice from a list of options."""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    
    while True:
        choice = input(f"Enter your choice (1-{len(options)}): ").strip()
        if not choice and default is not None:
            return default
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print(f"Invalid choice. Please enter a number between 1 and {len(options)}.")

def main():
    # Initialize the orchestrator
    orchestrator = TwoStageRetrievalOrchestrator("http://localhost:8983/solr")
    
    # Available datasets
    datasets = ["coco", "flickr"]
    
    # Available models
    models = ["clip", "minilm", "uniir", "flava"]
    
    # Core types
    core_types = ["text", "image"]
    
    print("="*60)
    print("Two-Stage Retrieval System")
    print("="*60)
    
    while True:
        try:
            # Step 1: Choose dataset
            dataset = get_user_choice("Select a dataset:", datasets)
            
            # Step 2: First stage configuration
            print(f"\n--- First Stage Configuration (Top 100) ---")
            stage1_model = get_user_choice("Select first stage model:", models)
            stage1_core_type = get_user_choice("Select first stage core type:", core_types, "text")
            stage1_core = f"{stage1_model}_{dataset}_{stage1_core_type}"
            
            # Step 3: Second stage configuration
            print(f"\n--- Second Stage Configuration (Top 10) ---")
            stage2_model = get_user_choice("Select second stage model:", models)
            stage2_core_type = get_user_choice("Select second stage core type:", core_types, "text")
            stage2_core = f"{stage2_model}_{dataset}_{stage2_core_type}"
            
            # Display the selected pipeline
            print(f"\nSelected Pipeline:")
            print(f"  Stage 1: {stage1_model} -> {stage1_core}")
            print(f"  Stage 2: {stage2_model} -> {stage2_core}")
            
            # Get query from user
            query = input("\nEnter your search query (or 'back' to reconfigure, 'exit' to quit): ").strip()
            
            if query.lower() == 'exit':
                break
            if query.lower() == 'back':
                continue
                
            # Execute the chosen pipeline
            result = orchestrator.execute_two_stage_pipeline(
                query_text=query,
                stage1_config=(stage1_model, stage1_core),
                stage2_config=(stage2_model, stage2_core),
                stage1_k=100,  # Get top 100 from first stage
                stage2_k=10    # Get top 10 from second stage
            )

                        # Print the results
            orchestrator.print_results(result)
            
            # Show option to view more Stage 1 results
            # if "stage1_results" in result and len(result["stage1_results"]) > 10:
            #     show_more = input("\nWould you like to see more results from Stage 1? (y/n): ").strip().lower()
            #     if show_more == 'y':
            #         print(f"\nALL {len(result['stage1_results'])} RESULTS FROM STAGE 1:")
            #         for i, doc in enumerate(result["stage1_results"], 1):
            #             print(f"  {i}. {doc.get('image_path', 'N/A')} (Score: {doc.get('score', 'N/A'):.6f})")
            
            # Ask if user wants to try another query with same configuration
            another = input("\nWould you like to try another query with the same configuration? (y/n): ").strip().lower()
            if another != 'y':
                print("Returning to configuration...")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()