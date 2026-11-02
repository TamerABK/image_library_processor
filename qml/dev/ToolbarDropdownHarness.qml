import QtQuick
import QtQuick.Controls
import "../components" as Components

ApplicationWindow {
    visible: true
    width: 500
    height: 350
    property string selectedOption: ""
    Components.ToolbarDropdownButton {
        objectName: "standaloneDropdown"
        x: 100
        y: 100
        options: ["First", "Second"]
        onOptionSelected: function(option) { selectedOption = option }
    }
}
