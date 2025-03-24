for app in $(az containerapp list --resource-group liver-tumor-rg --query "[].name" -o tsv); do
  echo "Deleting container app: $app"
  az containerapp delete --name $app --resource-group liver-tumor-rg --yes
done