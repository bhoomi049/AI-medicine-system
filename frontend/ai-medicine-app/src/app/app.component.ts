import { Component } from '@angular/core';
import { PatientService } from './services/patient.service';
import { CommonModule } from '@angular/common'; // For *ngIf
import { FormsModule } from '@angular/forms';   // For [(ngModel)]

@Component({
  selector: 'app-root',
  standalone: true, // Standalone component
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
  imports: [CommonModule, FormsModule] // Import required modules
})
export class AppComponent {
  title = 'AI Medicine System';

  // Here, 'disease' is being used as an input for symptoms.
  // You might consider renaming it to 'symptoms' for clarity.
  patient = { name: '', age: 0, disease: '' };
  treatment: any = null; // Will store the treatment recommendation
  isLoading = false;     // For showing a loading indicator

  constructor(private patientService: PatientService) {}

  addPatient() {
    this.isLoading = true;
    this.patientService.addPatient(this.patient)
      .then(response => {
        alert(response.data.message);
        this.isLoading = false;
      })
      .catch(error => {
        console.error("Error adding patient:", error);
        this.isLoading = false;
      });
  }

  predictTreatment() {
    this.isLoading = true;
    // Send an object with a 'symptoms' array to the backend.
    // Here we assume that the input in the "disease" field is a comma-separated list of symptoms.
    const symptomsArray = this.patient.disease.split(',').map(s => s.trim());
    const features = { symptoms: symptomsArray };

    this.patientService.getPrediction(features)
      .then(response => {
        console.log("API response:", response.data);
        this.treatment = response.data.treatment;
        this.isLoading = false;
      })
      .catch(error => {
        console.error("Prediction error:", error);
        this.isLoading = false;
      });
  }
}
