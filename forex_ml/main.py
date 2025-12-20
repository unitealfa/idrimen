"""
Forex ML Prediction System - Main Entry Point
Interactive menu to Train or Predict
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def display_menu():
    """Display the main menu"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print()
    print("╔" + "═"*56 + "╗")
    print("║" + " "*15 + "FOREX ML PREDICTION SYSTEM" + " "*15 + "║")
    print("║" + " "*18 + "LSTM Multi-Output Model" + " "*15 + "║")
    print("╠" + "═"*56 + "╣")
    print("║" + " "*56 + "║")
    print("║   1. TRAIN   - Entraîner le modèle avec les données   ║")
    print("║               existantes (forex_live_data.csv)        ║")
    print("║" + " "*56 + "║")
    print("║   2. PREDICT - Mode prédiction en temps réel          ║")
    print("║               (validation toutes les 10 minutes)      ║")
    print("║" + " "*56 + "║")
    print("║   0. QUIT    - Quitter le programme                   ║")
    print("║" + " "*56 + "║")
    print("╚" + "═"*56 + "╝")
    print()


def main():
    """Main entry point"""
    
    # Check TensorFlow
    print("Initialisation...")
    try:
        import tensorflow as tf
        print(f"TensorFlow {tf.__version__} détecté")
        
        # Disable GPU warnings
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    except ImportError:
        print("❌ TensorFlow non installé!")
        print("   Installez-le avec: pip install tensorflow")
        input("\nAppuyez sur Entrée pour quitter...")
        sys.exit(1)
    
    while True:
        display_menu()
        choice = input("Votre choix: ").strip()
        
        if choice == "1":
            print("\n" + "="*60)
            print("   Chargement du module d'entraînement...")
            print("="*60)
            
            try:
                from train import run_training
                run_training()
            except Exception as e:
                print(f"\n❌ Erreur: {e}")
                import traceback
                traceback.print_exc()
                input("\nAppuyez sur Entrée pour continuer...")
                
        elif choice == "2":
            print("\n" + "="*60)
            print("   Chargement du module de prédiction...")
            print("="*60)
            
            try:
                from predict_live import run_prediction_loop
                run_prediction_loop()
            except Exception as e:
                print(f"\n❌ Erreur: {e}")
                import traceback
                traceback.print_exc()
                input("\nAppuyez sur Entrée pour continuer...")
                
        elif choice == "0":
            print("\n👋 Au revoir!")
            sys.exit(0)
            
        else:
            print("\n⚠️ Choix invalide! Entrez 0, 1 ou 2.")
            input("\nAppuyez sur Entrée pour continuer...")


if __name__ == "__main__":
    main()
