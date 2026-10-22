pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components
import "../theme" as Room36Theme

Item {
    id: root
    objectName: "unknownPeoplePage"
    property var people: null
    property var inspector: null
    property var favorites: null
    property var exporter: null
    property string viewSize: "Medium"
    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    readonly property bool detailOpen: people !== null && Boolean(people.selectedUnknown.key)
    function openNaming() {
        mode.currentIndex = 0
        nameField.text = ""
        namingDialog.open()
        nameField.forceActiveFocus()
    }

    ColumnLayout {
        anchors.fill: parent
        visible: !root.detailOpen
        spacing: 12
        RowLayout {
            Layout.fillWidth: true
            Components.ToolbarDropdownButton {
                objectName: "unknownPeopleFilter"
                label: root.people ? root.people.unknownModel.clusterFilter : "Needs review"
                options: ["Needs review", "Ignored", "All"]
                onOptionSelected: function(value) { if (root.people) root.people.unknownModel.setFilter(value) }
                Accessible.name: "Filter unknown people: " + label
            }
            Components.ToolbarDropdownButton {
                objectName: "unknownPeopleSort"
                label: root.people ? root.people.unknownModel.sortOrder : "Largest clusters"
                options: ["Largest clusters", "Smallest clusters"]
                onOptionSelected: function(value) { if (root.people) root.people.unknownModel.setSort(value) }
                Accessible.name: "Sort unknown people: " + label
            }
            Item { Layout.fillWidth: true }
        }
        Text {
            objectName: "unknownPeopleStateText"
            Layout.fillWidth: true
            visible: root.people === null || root.people.state !== "COMPLETE" || root.people.unknownModel.visibleCount === 0
            text: !root.people || root.people.state !== "COMPLETE"
                ? (root.people && root.people.state === "FAILED"
                    ? "People analysis is unavailable. " + root.people.stageError
                    : "Processing people…")
                : root.people.unknownModel.clusterFilter === "Needs review"
                    ? "No unknown people need review." : "No clusters match this filter."
            color: theme.mainBlue
            wrapMode: Text.WordWrap
            font.pixelSize: typography.body
        }
        Components.PillButton {
            objectName: "unknownPeopleRetry"
            variant: "accent"
            visible: root.people !== null && root.people.state === "FAILED"
            text: "Retry people analysis"
            onClicked: if (root.people) root.people.retryProcessing()
            Accessible.name: "Retry project processing"
        }
        Components.PeopleGrid {
            Layout.fillWidth: true
            Layout.fillHeight: true
            visible: root.people !== null && root.people.state === "COMPLETE"
            peopleModel: root.people ? root.people.unknownModel : null
            viewSize: root.viewSize
            known: false
            onPersonOpened: function(key) { if (root.people) root.people.openUnknown(key) }
            onNameRequested: function(key) {
                if (root.people) { root.people.openUnknown(key); root.openNaming() }
            }
            onIgnoreRequested: function(key, ignored) {
                if (root.people) root.people.setIgnored(key, ignored)
            }
        }
    }
    Components.PeopleDetailPhotos {
        anchors.fill: parent
        visible: root.detailOpen
        person: root.people ? root.people.selectedUnknown : ({})
        title: "Unknown person"
        photoModel: root.people ? root.people.detailPhotoModel : null
        viewSize: root.viewSize
        inspector: root.inspector
        favoritesController: root.favorites
        exporter: root.exporter
        onBackRequested: if (root.people) root.people.back()
        actions: RowLayout {
            Components.PillButton {
                objectName: "unknownDetailName"
                variant: "filled"
                text: "Name person"
                enabled: root.people !== null && !root.people.busy
                onClicked: root.openNaming()
                Accessible.name: "Name or assign this unknown person"
            }
            Components.PillButton {
                objectName: "unknownDetailIgnore"
                variant: "filled"
                text: root.people && root.people.selectedUnknown.ignored ? "Restore" : "Ignore"
                enabled: root.people !== null && !root.people.busy
                onClicked: if (root.people)
                    root.people.setIgnored(root.people.selectedUnknown.key, !root.people.selectedUnknown.ignored)
                Accessible.name: text + " this unknown person"
            }
        }
    }
    Text {
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        visible: root.people !== null && (root.people.busy || root.people.error.length > 0)
        text: root.people ? (root.people.busy ? root.people.busyMessage : root.people.error) : ""
        color: theme.mainBlue
        font.pixelSize: typography.body
        Accessible.name: text
    }
    Dialog {
        id: namingDialog
        objectName: "unknownNamingDialog"
        parent: Overlay.overlay
        anchors.centerIn: parent
        modal: true
        title: "Name unknown person"
        width: Math.min(420, root.width - 40)
        standardButtons: Dialog.NoButton
        background: Rectangle {
            radius: 20
            color: theme.panelFill
            border.width: 1
            border.color: theme.separator
        }
        ColumnLayout {
            width: namingDialog.availableWidth
            spacing: 12
            Components.RoundedComboBox {
                id: mode
                objectName: "unknownNamingMode"
                Layout.fillWidth: true
                model: ["Create new person", "Assign to existing person"]
                Accessible.name: "Person assignment mode"
            }
            TextField {
                id: nameField
                objectName: "unknownPersonName"
                Layout.fillWidth: true
                visible: mode.currentIndex === 0
                placeholderText: "Person name"
                maximumLength: 100
                Accessible.name: "New person name"
                onAccepted: confirmButton.clicked()
                implicitHeight: 42
                background: Rectangle {
                    radius: height / 2
                    color: theme.panelRaisedFill
                    border.width: nameField.activeFocus ? 2 : 1
                    border.color: nameField.activeFocus ? theme.focusRing : theme.separator
                }
            }
            Components.RoundedComboBox {
                id: existing
                objectName: "unknownExistingPerson"
                Layout.fillWidth: true
                visible: mode.currentIndex === 1
                model: root.people ? root.people.knownOptions : []
                textRole: "name"
                valueRole: "personId"
                Accessible.name: "Choose an existing known person"
            }
            Text {
                visible: root.people !== null && root.people.suggestion.name !== undefined
                text: root.people ? "Suggested: " + root.people.suggestion.name
                    + " (similarity " + root.people.suggestion.score + ")" : ""
                color: theme.mutedText
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
            Text {
                visible: root.people !== null && root.people.error.length > 0
                text: root.people ? root.people.error : ""
                color: theme.mainBlue
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
                Accessible.name: text
            }
            RowLayout {
                Layout.alignment: Qt.AlignRight
                Components.PillButton {
                    objectName: "unknownCancelName"
                    variant: "translucent"
                    text: "Cancel"
                    onClicked: namingDialog.close()
                    Accessible.name: "Cancel naming person"
                }
                Components.PillButton {
                    id: confirmButton
                    objectName: "unknownConfirmName"
                    variant: "accent"
                    text: mode.currentIndex === 0 ? "Create person" : "Assign person"
                    enabled: root.people !== null && !root.people.busy
                        && (mode.currentIndex === 0 || existing.count > 0)
                    onClicked: {
                        if (!root.people) return
                        const accepted = mode.currentIndex === 0
                            ? root.people.nameNew(nameField.text)
                            : root.people.assignExisting(existing.currentValue)
                        if (accepted) namingDialog.close()
                    }
                    Accessible.name: text
                }
            }
        }
    }
}
